#!/usr/bin/env python3
#
# SpMV paginado por filas (CSR empaquetado) usando objectFifo + npu_dma_memcpy_nd
#
# Layout por página de A:
#   [ rowptr( RPB+1 ) | colidx( page_nnz_max ) | vals( page_nnz_max ) ]
#
# X e Y entran enteros en memoria (no se paginan).
#
# (c) 2025 AMD Inc. / adaptado
import argparse
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_


def my_spmv(
    dev: str,
    rows: int,
    cols: int,
    nnz: int,
    page_rows_max: int,
    page_nnz_max: int,
):
    """
    Genera el módulo MLIR para un SpMV CSR paginado:
      y = A * x

    - A se pasa como un gran vector A_COMBINED_ALL de longitud TOTAL_COMBINED_LEN,
      formado por NUM_PAGES páginas consecutivas.
    - Cada página: (page_rows_max+1) + 2*page_nnz_max elementos i32.
    - X e Y se pasan como vectores densos de tamaño cols y rows respectivamente.
    """

    # -------------------- parámetros de paginado --------------------
    RPB = page_rows_max                     # filas por página (Rows Per Block)
    MAX_NNZ_PER_PAGE = page_nnz_max         # NNZ máximo por página

    # Largo de una página "combinada" (rowptr|colidx|vals)
    COMBINED_PAGE_LEN = (RPB + 1) + 2 * MAX_NNZ_PER_PAGE

    # Cantidad de páginas necesarias para cubrir 'rows'
    NUM_PAGES = (rows + RPB - 1) // RPB     # ceil(rows / RPB)

    # Longitud total del buffer combinado de A
    TOTAL_COMBINED_LEN = NUM_PAGES * COMBINED_PAGE_LEN

    # Tipos de datos (por ahora fijos a i32 in/out como tu kernel)
    dtype_in = np.dtype[np.int32]
    dtype_out = np.dtype[np.int32]
    dtype_in_str = "i32"
    dtype_out_str = "i32"

    with mlir_mod_ctx() as ctx:
        # -------------------- seleccionar device --------------------
        if dev == "npu":
            # NPU1: usamos el arreglo completo, pero solamente una columna (0)
            dev_ty = AIEDevice.npu1
        else:
            # NPU2: idem, usamos la columna 0
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            # -------------------- tipos en L1/L2 --------------------
            # Una página combinada de A (CSR empaquetado)
            A_page_ty = np.ndarray[(COMBINED_PAGE_LEN,), dtype_in]
            # Vector X completo
            X_ty = np.ndarray[(cols,), dtype_in]
            # Chunk de Y (a lo sumo RPB filas)
            Y_page_ty = np.ndarray[(RPB,), dtype_out]

            # AIE Core Function declarations (kernels C++)
            zero = external_func(f"zero_{dtype_out_str}", inputs=[Y_page_ty])
            spmv = external_func(
                f"spmv_{dtype_in_str}_{dtype_out_str}",
                inputs=[A_page_ty, X_ty, Y_page_ty],
            )

            # -------------------- tiles --------------------
            # Usamos la columna 0: shim (0,0), memtile (0,1), core (0,2)
            ShimTile0 = tile(0, 0)
            MemTile0 = tile(0, 1)
            CoreTile0 = tile(0, 2)

            # -------------------- ObjectFifos --------------------
            fifo_depth = 2

            # A: host -> shim -> mem -> core
            A_L3L2 = object_fifo(
                "A_L3L2",
                ShimTile0,   # productor DMA (shim)
                MemTile0,    # consumidor L2
                fifo_depth,
                A_page_ty,
            )
            A_L2L1 = object_fifo(
                "A_L2L1",
                MemTile0,    # productor L2
                CoreTile0,   # consumidor L1
                fifo_depth,
                A_page_ty,
            )
            object_fifo_link(A_L3L2, A_L2L1)

            # X: host -> shim -> core (broadcast, pero ahora solo 1 core)
            X_fifo = object_fifo(
                "X_fifo",
                ShimTile0,   # productor
                CoreTile0,   # consumidor
                fifo_depth,
                X_ty,
            )

            # Y: core -> shim (un chunk de RPB filas)
            Y_fifo = object_fifo(
                "Y_fifo",
                CoreTile0,   # productor
                ShimTile0,   # consumidor
                fifo_depth,
                Y_page_ty,
            )

            # -------------------- programa del core --------------------
            @core(CoreTile0, "spmv_sparse.o", stack_size=0x400)
            def core_body():
                # Cargamos X una sola vez y la dejamos residente en L1
                x_tile = X_fifo.acquire(ObjectFifoPort.Consume, 1)
                y_page = Y_fifo.acquire(ObjectFifoPort.Produce, 1)

                # Loop infinito: cada iteración procesa UNA página de A -> Y_chunk
                for _ in range_(0xFFFFFFFF):
                    # Página CSR de A para este core
                    a_page = A_L2L1.acquire(ObjectFifoPort.Consume, 1)
                    # Chunk de salida Y

                    # Inicializamos Y_chunk
                    zero(y_page)

                    # SpMV local: A_page * X -> Y_page
                    spmv(a_page, x_tile, y_page)

                    # Liberamos A y Y (X queda fijo en L1)
                    A_L2L1.release(ObjectFifoPort.Consume, 1)
                    
                    Y_fifo.release(ObjectFifoPort.Produce, 1)

                # Si el core alguna vez sale del bucle, liberaría X aquí
                X_fifo.release(ObjectFifoPort.Consume, 1)

            # -------------------- runtime_sequence --------------------
            # Tipos L3 (host-side)
            A_all_ty = np.ndarray[(TOTAL_COMBINED_LEN,), dtype_in]
            X_all_ty = X_ty
            Y_all_ty = np.ndarray[(rows,), dtype_out]

            @runtime_sequence(A_all_ty, X_all_ty, Y_all_ty)
            def sequence(A_COMBINED_ALL, X_VEC, Y_VEC):
                # --- 1) Enviar X una sola vez a X_fifo ---
                npu_dma_memcpy_nd(
                    metadata=X_fifo,
                    bd_id=2,                # id arbitrario, igual que ejemplo matvec
                    mem=X_VEC,
                    sizes=[1, 1, 1, cols],
                    strides=[0, 0, 0, 1],
                )
# Y: core -> Y_fifo -> Y_VEC[row_start : row_start+row_len]
                npu_dma_memcpy_nd(
                    metadata=Y_fifo,
                    bd_id=0,
                    mem=Y_VEC,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, 1, 1, rows],
                    strides=[0, 0, 0, 1],
                )
                # --- 2) Para cada página de A, mandar página y recoger chunk de Y ---
                for p in range(NUM_PAGES):
                    page_offset = p * COMBINED_PAGE_LEN
                    row_start = p * RPB
                    row_len = min(RPB, rows - row_start)

                    # A: host -> A_L3L2 (que está linkeado a A_L2L1 -> core)
                    npu_dma_memcpy_nd(
                        metadata=A_L3L2,
                        bd_id=1,
                        mem=A_COMBINED_ALL,
                        offsets=[0, 0, 0, page_offset],
                        sizes=[1, 1, 1, COMBINED_PAGE_LEN],
                        strides=[0, 0, 0, 1],
                    )

                    

                # Esperar a que terminen todos los DMA asociados a Y_fifo
                dma_wait(Y_fifo)

        # imprimir módulo MLIR resultante
        print(ctx.module)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="AIE SpMV MLIR Design (paginado CSR por filas, 1 core)",
        description="Genera MLIR para un diseño SpMV CSR paginado usando npu_dma_memcpy_nd.",
    )

    # Dispositivo: npu1 vs npu2
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu2")

    # Tamaño de la matriz
    argparser.add_argument(
        "-R",
        "--rows",
        type=int,
        default=512,
        help="Número de filas de la matriz (y longitud de y).",
    )
    argparser.add_argument(
        "-C",
        "--cols",
        type=int,
        default=512,
        help="Número de columnas de la matriz (y longitud de x).",
    )
    argparser.add_argument(
        "--nnz",
        type=int,
        default=2048,
        help="Total de elementos no nulos de la matriz (se usa sólo para debug / checks).",
    )

    # Paginado de A
    argparser.add_argument(
        "--page-rows-max",
        type=int,
        default=128,
        help="Máximo de filas por página de A (R_local / RPB).",
    )
    argparser.add_argument(
        "--page-nnz-max",
        type=int,
        default=1024,
        help="Máximo de nnz por página de A (NNZ_local).",
    )

    args, _ = argparser.parse_known_args()

    my_spmv(
        dev=args.dev,
        rows=args.rows,
        cols=args.cols,
        nnz=args.nnz,
        page_rows_max=args.page_rows_max,
        page_nnz_max=args.page_nnz_max,
    )

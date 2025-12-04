#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023-2025 AMD Inc.
import argparse
import numpy as np

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessSequence
from aie.iron import str_to_dtype


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE SpMV MLIR Design (paginado por filas)",
        description="Emite MLIR para un diseño SpMV multi-core con A paginado",
    )

    # Dispositivo: npu1 vs npu2
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu2")

    # Tamaño de la matriz
    argparser.add_argument(
        "-R", "--rows",
        type=int,
        default=512,
        help="Número de filas de la matriz (y longitud de x,y).",
    )
    argparser.add_argument(
        "-C", "--cols",
        type=int,
        default=512,
        help="Número de columnas de la matriz (en CSR normalmente R=C, pero no lo asumimos).",
    )
    argparser.add_argument(
        "--nnz",
        type=int,
        default=2048,
        help="Total de elementos no nulos de la matriz.",
    )

    # Paginado de A
    argparser.add_argument(
        "--page-rows-max",
        type=int,
        default=128,
        help="Máximo de filas por página de A (R_local).",
    )
    argparser.add_argument(
        "--page-nnz-max",
        type=int,
        default=1024,
        help="Máximo de nnz por página de A (NNZ_local).",
    )

    # Cantidad de cores (columnas de AIE que vas a usar)
    argparser.add_argument(
        "--n-aie-cols",
        type=int,
        choices=[1, 2, 4, 8],
        default=4,
        help="Cantidad de columnas AIE (cores SpMV) que se usan.",
    )

    # Tipos de datos
    argparser.add_argument(
        "--dtype_in",
        type=str,
        choices=["bf16", "i8", "i16", "i32"],
        default="i32",
        help="Tipo de entrada (valores de A y x).",
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "i32", "f32"],
        default="i32",
        help="Tipo de salida (y).",
    )

    # Trace / taps como en el ejemplo de matmul
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()

    with mlir_mod_ctx() as ctx:
        maybe_taps = my_spmv(
            dev=args.dev,
            rows=args.rows,
            cols=args.cols,
            nnz=args.nnz,
            page_rows_max=args.page_rows_max,
            page_nnz_max=args.page_nnz_max,
            n_aie_cols=args.n_aie_cols,
            dtype_in=str_to_dtype(args.dtype_in),
            dtype_out=str_to_dtype(args.dtype_out),
            trace_size=args.trace_size,
        )
        print(ctx.module)

    @device(dev_ty)
    def device_body():
        # ----- TAMAÑOS DE L2/L1 -----
        A_tile_width = (page_rows_max + 1) + 2 * page_nnz_max

        # Tipos en L2 (shim <-> mem)
        A_l2_ty = np.ndarray[(A_tile_width,), dtype_in]      # página de A (rowptr|colidx|vals)
        X_l2_ty = np.ndarray[(cols,),        dtype_in]       # X completo
        C_l2_ty = np.ndarray[(page_rows_max,), dtype_out]    # chunk de Y (máx filas por página)

        # Por ahora, L1 == L2
        A_l1_ty = A_l2_ty
        X_l1_ty = X_l2_ty
        C_l1_ty = C_l2_ty

        # --- strings para nombres de kernels externos ---
        dtype_map = {
            np.dtype("int8"):  "i8",
            np.dtype("int16"): "i16",
            np.dtype("int32"): "i32",
        }
        dtype_in_str  = dtype_map[dtype_in]
        dtype_out_str = dtype_map[dtype_out]

        # AIE Core Function declarations
        zero = external_func(f"zero_{dtype_out_str}", inputs=[C_l1_ty])
        spmv_func_name = f"spmv_{dtype_in_str}_{dtype_out_str}"
        spmv = external_func(
            spmv_func_name,
            inputs=[A_l1_ty, X_l1_ty, C_l1_ty],
        )

        # Tile declarations as tile[row][col]
        tiles = [
            [tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)
        ]
        shim_tiles = tiles[0]
        mem_tiles  = tiles[1]
        core_tiles = tiles[2:]   # 4 filas: rows 2,3,4,5

        # De momento solo soportamos n_aie_cols <= 4
        n_shim_mem_A = n_aie_cols

        # --------------- ObjectFifos A ---------------
        A_l3l2_fifos = [None] * n_shim_mem_A
        A_l2l1_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]

        # L3 -> L2: shim -> mem
        for col in range(n_shim_mem_A):
            A_l3l2_fifos[col] = object_fifo(
                f"A_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                A_l2_ty,
            )

        # L2 -> L1: mem -> cores
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                A_l2l1_fifos[row][col] = object_fifo(
                    f"A_L2L1_{row}_{col}",
                    mem_tiles[col],
                    core_tiles[row][col],
                    fifo_depth,
                    A_l1_ty,
                )

        # link: cada columna distribuye sus páginas de A a las 4 filas de cores
        for col in range(n_shim_mem_A):
            object_fifo_link(
                A_l3l2_fifos[col],
                [A_l2l1_fifos[row][col] for row in range(n_aie_rows)],
            )

        # --------------- ObjectFifos X (broadcast) ---------------
        shim_for_x = shim_tiles[0]

        # broadcast L3 -> L2: un shim a todos los mem tiles
        X_L3L2 = object_fifo(
            "X_L3L2_broadcast",
            shim_for_x,
            [mem_tiles[c] for c in range(n_aie_cols)],
            fifo_depth,
            X_l2_ty,
        )

        # broadcast L2 -> L1: cada mem tile a los cores de su columna
        X_L2L1 = []
        for col in range(n_aie_cols):
            X_L2L1.append(
                object_fifo(
                    f"X_L2L1_broadcast_col{col}",
                    mem_tiles[col],
                    [core_tiles[row][col] for row in range(n_aie_rows)],
                    fifo_depth,
                    X_l1_ty,
                )
            )

        # --------------- ObjectFifos C (salida) ---------------
        C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]

        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                # L1 -> L2: core -> mem
                C_l1l2_fifos[row][col] = object_fifo(
                    f"C_L1L2_{row}_{col}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    fifo_depth,
                    C_l1_ty,
                )
                # L2 -> L3: mem -> shim (uno por core, más simple por ahora)
                C_l2l3_fifos[row][col] = object_fifo(
                    f"C_L2L3_{row}_{col}",
                    mem_tiles[col],
                    shim_tiles[col],
                    fifo_depth,
                    C_l2_ty,
                )
                object_fifo_link(
                    C_l1l2_fifos[row][col],
                    C_l2l3_fifos[row][col],
                )

        # TODO (próximo paso):
        #  - Definir @core(...) por (row,col) usando spmv(...)
        #  - Definir runtime_sequence que:
        #      * hace DMA de páginas de A -> A_L3L2
        #      * hace DMA de X -> X_L3L2
        #      * drena C_L2L3_* hacia Y global, con offsets por core/página
        # --------------- Set up compute tiles (cores) ---------------
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                # Alias locales para que el cierre de Python no se rompa
                of_A = A_l2l1_fifos[row][col]
                of_X = X_L2L1[col]
                of_C = C_l1l2_fifos[row][col]

                @core(core_tiles[row][col], "spmv_sparse.o", stack_size=0x400)
                def core_body():
                    # Pre-cargamos X una sola vez en este core.
                    # El host tiene que haber enviado al menos 1 tile de X por columna.
                    x_tile = of_X.acquire(ObjectFifoPort.Consume, 1)

                    # Loop infinito: cada iteración procesa UNA página de A -> C
                    for _ in range_(0xFFFFFFFF):
                        # Página de A para este core
                        a_tile = of_A.acquire(ObjectFifoPort.Consume, 1)
                        # Chunk de salida C (hasta page_rows_max filas)
                        c_tile = of_C.acquire(ObjectFifoPort.Produce, 1)

                        # Inicializamos C (si tenés kernel zero_*)
                        zero(c_tile)

                        # SpMV local: A_page * X -> C_page
                        spmv(a_tile, x_tile, c_tile)

                        # Liberamos A y C (X queda residente en L1)
                        of_A.release(ObjectFifoPort.Consume, 1)
                        of_C.release(ObjectFifoPort.Produce, 1)

                    # Nota: no hacemos release de x_tile a propósito,
                    # la idea es que X quede "pinneado" en el core.
                    of_X.release(ObjectFifoPort.Consume, 1)

        RPB = page_rows_max
        COMBINED_PAGE_LEN = (page_rows_max + 1) + 2 * page_nnz_max
        NUM_PAGES = (rows + RPB - 1) // RPB   # ceildiv
        TOTAL_COMBINED_LEN = NUM_PAGES * COMBINED_PAGE_LEN

        A_all_ty = np.ndarray[(TOTAL_COMBINED_LEN,), np.dtype[dtype_in]]
        X_ty     = np.ndarray[(cols,),              np.dtype[dtype_in]]
        Y_ty     = np.ndarray[(rows,),             np.dtype[dtype_out]]

        @runtime_sequence(A_all_ty, X_ty, Y_ty)
        def sequence(A_COMBINED_ALL, X_VEC, Y_VEC):
            x_task = shim_dma_single_bd_task(
                X_L3L2, X_VEC,
                sizes=[1, 1, 1, cols],
                issue_token=True,
                offset=0,
            )

            num_chunks = (NUM_PAGES + MAX_PAGES_IN_FLIGHT - 1) // MAX_PAGES_IN_FLIGHT

            for chunk in range(num_chunks):
                first_page = chunk * MAX_PAGES_IN_FLIGHT
                last_page  = min(first_page + MAX_PAGES_IN_FLIGHT, NUM_PAGES)

                in_tasks  = []
                out_tasks = []

                # X solo la primera vez
                if chunk == 0:
                    in_tasks.append(x_task)

                # ---- A: L3 -> L2, páginas de este chunk ----
                for p in range(first_page, last_page):
                    page_offset = p * COMBINED_PAGE_LEN
                    col = p % n_aie_cols

                    in_tasks.append(
                        shim_dma_single_bd_task(
                            A_l3l2_fifos[col],
                            A_COMBINED_ALL,
                            sizes=[1, 1, 1, COMBINED_PAGE_LEN],
                            issue_token=True,
                            offset=page_offset,
                        )
                    )

                # ---- C: L2 -> L3, escribir en Y ----
                for p in range(first_page, last_page):
                    row_start = p * RPB
                    row_len   = min(RPB, rows - row_start)

                    col = p % n_aie_cols
                    row_core = 0  # de momento mandás todo al row 0 de esa col

                    out_tasks.append(
                        shim_dma_single_bd_task(
                            C_l2l3_fifos[row_core][col],
                            Y_VEC,
                            sizes=[1, 1, 1, row_len],
                            issue_token=True,
                            offset=row_start,
                        )
                    )

                # ---- Lanzar y esperar este chunk ----
                dma_start_task(*in_tasks, *out_tasks)
                dma_await_task(*out_tasks)
                dma_free_task(*in_tasks)
if __name__ == "__main__":
    main()

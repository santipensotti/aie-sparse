import argparse
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.iron.dtype import str_to_dtype

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, choices=["npu", "npu2"], default="npu")
    ap.add_argument("-nnz", type=int, default=16)
    ap.add_argument("-N", type=int, default=8)
    ap.add_argument("-M", type=int, default=8)
    ap.add_argument("--dtype_in", type=str, choices=["i16", "i8", "bf16"], default="i16")
    ap.add_argument("--dtype_out", type=str, choices=["i32", "f32", "i16", "i8", "bf16"], default="i32")
    ap.add_argument("--trace_size", type=int, default=0)
    args = ap.parse_args()

    with mlir_mod_ctx() as ctx:
        spmv(
            dev=args.device,
            NNZ=args.nnz,
            N=args.N,
            M=args.M,
            dtype_in_str=args.dtype_in,
            dtype_out_str=args.dtype_out,
            trace_size=args.trace_size,
        )
        print(ctx.module)

def spmv(dev, NNZ, N, M, dtype_in_str, dtype_out_str, trace_size):
    # Tipos (usamos i32 para todo en este esqueleto)
    vals_ty_np   = np.int32
    x_ty_np      = np.int32
    y_ty_np      = np.int32
    rowptr_ty_np = np.int32
    colidx_ty_np = np.int32
    off_row = 0
    off_col = off_row + (M + 1)
    off_val = off_col + NNZ
    off_x   = off_val + NNZ
    total_elems = off_x + N
    combined_ty = np.ndarray[(total_elems,), np.dtype[np.int32]]
     
    # Memrefs (estilo numpy de IRON)
    rowptr_ty = np.ndarray[(M + 1,), np.dtype[np.int32]]
    colidx_ty = np.ndarray[(NNZ,), np.dtype[np.int32]]
    vals_ty   = np.ndarray[(NNZ,), np.dtype[np.int32]]
    x_ty      = np.ndarray[(N,),   np.dtype[np.int32]]
    y_ty      = np.ndarray[(M,),   np.dtype[np.int32]]

    # Buffer combinado: [rowptr | colidx | vals | x]
    off_row = 0
    off_col = off_row + (M + 1)
    off_val = off_col + NNZ
    off_x   = off_val + NNZ
    total_elems = off_x + N
    combined_ty = np.ndarray[(total_elems,), np.dtype[np.int32]]

    # Dispositivo
    dev_ty = AIEDevice.npu1_1col if dev == "npu" else AIEDevice.npu2

    @device(dev_ty)
    def device_body():
        # Tiles
        shim   = tile(0, 0)
        t_core = tile(0, 2)

        # Kernel externo (C/C++): spmv_sparse(int32*, int32*, int32*, int32*, int32*)
        spmv_fn = external_func(
            "spmv_sparse",
            inputs=[combined_ty, y_ty],
        )

        # FIFOs: 1 entrada (combinado) + 1 salida (y)
        fifo_in = object_fifo("fifo_in", shim,  t_core, 1, combined_ty)
        fifo_y  = object_fifo("fifo_y",  t_core, shim,  1, y_ty)

        @core(t_core, "spmv_core.o")
        def core_body():
            for _ in range_(1):
                # Obtener 4 tokens secuenciales del mismo FIFO (en el orden que los cargamos)
                row_pkt = fifo_in.acquire(ObjectFifoPort.Consume, 1)
                y_pkt   = fifo_y.acquire(ObjectFifoPort.Produce, 1)
        
                # Llamada al kernel C
                spmv_fn(row_pkt, y_pkt)

                # Liberar tokens
                fifo_in.release(ObjectFifoPort.Consume, 1)
                fifo_y.release(ObjectFifoPort.Produce, 1)

        # Runtime: un solo BO de entrada (COMBINED) con offsets
        @runtime_sequence(combined_ty, y_ty)
        def sequence(COMBINED, Y):
            # Copias 1D con offset dentro del mismo buffer
            t_row = shim_dma_single_bd_task(
                fifo_in, COMBINED, sizes=[1,1,1, 33], offset=0, issue_token=True
            )
           
            # Salida (core -> shim)
            t_y = shim_dma_single_bd_task(
                fifo_y, Y, sizes=[1,1,1, M], issue_token=True
            )

            # Lanzar DMAs (entrada secuenciada en el mismo canal)
            dma_start_task(t_row)
            dma_start_task(t_y)
            dma_await_task(t_y)


if __name__ == "__main__":
    main()

import numpy as np
import sys
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *

# --- device ---

dev = NPU2()if len(sys.argv) > 1 and sys.argv[1] == "npu2" else NPU1()

# --- problem sizes ---
tile_rows = 8

# --- kernel externo ---


def spmvSingleCore():
    with mlir_mod_ctx() as ctx:
        @device(dev)
        def device_body():
            # ---- FIFOs con endpoints explícitos ----
            ##             memC = ObjectFifo("memC", compute_tile2, mem_tile, 2, c_ty)

            # La idea es mem tile -> A una parte y mem tile -> B. Despues join A,B -> C -> mem tile


            ShimCol1 = tile(1, 0)
            MemCol1 = tile(1, 2)
            ComputeTileCol1 = tile(1, 3)

            ShimCol0 = tile(0, 0)
            MemCol0 = tile(0, 2)
            ComputeTileCol0 = tile(0, 3)

            # FIFOs
            of_in = object_fifo("objfifo_in", ShimCol1, MemCol1, 2, np.ndarray[(8,), np.dtype[np.int32]])
            of_in2 = object_fifo("objfifo_in2", ShimCol1, MemCol1, 2, np.ndarray[(8,), np.dtype[np.int32]])
            indexFifoCol1 = object_fifo("objfifo0", ShimCol1, ComputeTileCol1, 2, np.ndarray[(8,), np.dtype[np.int32]])
            rowFifoCol1 = object_fifo("objfifo1", ShimCol1, ComputeTileCol1, 2, np.ndarray[(8,), np.dtype[np.int32]])
            object_fifo_link(of_in, indexFifoCol1)
            object_fifo_link(of_in2, rowFifoCol1)

            # Fifos Col0
            colFifoCol2 = object_fifo("objfifoCol2", ShimCol0, ComputeTileCol0, 2, np.ndarray[(8,), np.dtype[np.int32]])
            xFifoCol2 = object_fifo("objfifoX2", ShimCol0, ComputeTileCol0, 2, np.ndarray[(8,), np.dtype[np.int32]])
            of_finCol1 = object_fifo("objfifo_fin1", ShimCol0, MemCol0, 2, np.ndarray[(8,), np.dtype[np.int32]])
            of_finCol2 = object_fifo("objfifo_fin2", ShimCol0, MemCol0, 2, np.ndarray[(8,), np.dtype[np.int32]])
            object_fifo_link(of_finCol1, colFifoCol2)
            object_fifo_link(of_finCol2, xFifoCol2)

            ## Aca tengo 
            """

            [ShimCol1] -> [ComputeTileCol1]
            [ShimCol1] -> [ComputeTileCol1]
            [ShimCol0] -> [ComputeTileCol0]
            [ShimCol0] -> [ComputeTileCol0]
            """
            # Falta hacer [ComputeTileCol1] -> [ComputeTileCol0]
            of_mid = object_fifo("objfifo_mid", ComputeTileCol1, ComputeTileCol0, 2, np.ndarray[(8,), np.dtype[np.int32]])
            ## 
            memOut = object_fifo("objfifo_output", ComputeTileCol0, MemCol0, 2, np.ndarray[(8,), np.dtype[np.int32]])
            outO = object_fifo("objfifo_out", MemCol0, ShimCol0, 2, np.ndarray[(8,), np.dtype[np.int32]])
            object_fifo_link(memOut, outO)

            tiles_to_trace = [ComputeTileCol0, ComputeTileCol1]

            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimCol0)


            spmv_fn = external_func(
                "matrix_vector_csr",
                inputs =[tile_rows, tile_rows, tile_rows, tile_rows, tile_rows],
            )


            @core(ComputeTileCol0, "matrix_vector_csr.o")
            def core_fn2():
                for _ in range_(1):
                    ind = indexFifoCol1.acquire(ObjectFifoPort.Consume, 1)
                    row = rowFifoCol1.acquire(ObjectFifoPort.Consume, 1)
                    col = colFifoCol2.acquire(ObjectFifoPort.Consume, 1)
                    x = xFifoCol2.acquire(ObjectFifoPort.Consume, 1)  
                    y = memOut.acquire(ObjectFifoPort.Produce, 1)
                    spmv_fn(ind, row, col, x, y)
                    indexFifoCol1.release(ObjectFifoPort.Consume, 1)
                    rowFifoCol1.release(ObjectFifoPort.Consume, 1)
                    colFifoCol2.release(ObjectFifoPort.Consume, 1)
                    xFifoCol2.release(ObjectFifoPort.Consume, 1)
                    memOut.release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(
                np.ndarray[(8,), np.dtype[np.int32]], # index pointer
                np.ndarray[(8,), np.dtype[np.int32]], # column indices
                np.ndarray[(8,), np.dtype[np.int32]], # values
                np.ndarray[(8,), np.dtype[np.int32]], # x
                np.ndarray[(8,), np.dtype[np.int32]], # output y
            )
            def sequence(inTensor_rowptr, inTensor_colidx, inTensor_vals, inTensor_x, outTensor_y):
                # Copio los inputs a las fifos de input del primer core

                npu_dma_memcpy_nd(
                    metadata=of_in, bd_id=1, mem=inTensor_rowptr, sizes=[1, 1, 1, 8]
                )

                npu_dma_memcpy_nd(
                    metadata=of_in2, bd_id=2, mem=inTensor_colidx, sizes=[1, 1, 1, 8]
                )
                # Copio los inputs a las fifos de input del segundo core
                npu_dma_memcpy_nd(
                    metadata=of_finCol1, bd_id=3, mem=inTensor_vals, sizes=[1, 1, 1, 8]
                )
                npu_dma_memcpy_nd(
                    metadata=of_finCol2, bd_id=4, mem=inTensor_x, sizes=[1, 1, 1, 8]
                )
                # Copio el output a la fifo de input
                npu_dma_memcpy_nd(
                    metadata=outO, bd_id=0, mem=outTensor_y, sizes=[1, 1, 1, 8]
                )
                dma_wait(outO)
                trace_utils.gen_trace_done_aie2(ShimCol0)


    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

spmvSingleCore()

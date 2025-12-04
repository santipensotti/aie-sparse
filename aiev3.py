# vector_scalar_add/vector_scalar_add_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
PROBLEM_SIZE = 512
NNZ_SIZE = 1536
AIE_COMBINED_WIDTH = 2*NNZ_SIZE + (PROBLEM_SIZE + 1)  # rowptr|colidx|vals

# --- device por defecto ---
dev = AIEDevice.npu1_1col
if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "npu2":
        dev = AIEDevice.npu2_1col
    else:
        raise ValueError(f"[ERROR] Device name {sys.argv[1]} is unknown")

def my_vector_bias_add():
    @device(dev)
    def device_body():
        combined_tile_ty = np.ndarray[(AIE_COMBINED_WIDTH,), np.dtype[np.int32]]
        all_data_ty      = np.ndarray[(PROBLEM_SIZE,),      np.dtype[np.int32]]

        ShimTile = tile(0, 0)
        MemTile  = tile(0, 1)
        ComputeTile2 = tile(0, 2)

        # FIFOs
        of_in0a = object_fifo("in0",  ShimTile, MemTile,      2, combined_tile_ty)
        of_in0b = object_fifo("in0b", MemTile,  ComputeTile2, 1, combined_tile_ty)
        object_fifo_link(of_in0a, of_in0b)

        of_in1  = object_fifo("in1",  ShimTile, ComputeTile2, 2, all_data_ty)   # x
        of_out1 = object_fifo("out1", ComputeTile2, ShimTile, 2, all_data_ty)   # y

        spmv_fn = external_func(
            "spmv_sparse",
            inputs=[combined_tile_ty, all_data_ty, all_data_ty],
        )

        @core(ComputeTile2, "spmv_core.o")
        def core_body():
            for _ in range_(sys.maxsize):
                combined_in = of_in0b.acquire(ObjectFifoPort.Consume, 1)
                x_in        = of_in1.acquire(ObjectFifoPort.Consume, 1)
                elem_out    = of_out1.acquire(ObjectFifoPort.Produce, 1)
                spmv_fn(combined_in, x_in, elem_out)
                of_in0b.release(ObjectFifoPort.Consume, 1)
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out1.release(ObjectFifoPort.Produce, 1)

        # IMPORTANT: tipos correctos en la signature
        @runtime_sequence(combined_tile_ty, all_data_ty, all_data_ty)
        def sequence(inCombinedTensor, inXTensor, outTensor):
            # Escribimos UN elemento de of_in0a con 3 BDs: rowptr | colidx | vals
            # tamaños
            # tengo que hacer como hacen en matrix multiply el tema de los bd que usan como maximo 5
            bd_rowptr = shim_dma_single_bd_task(of_in0a, inCombinedTensor,
                sizes=[1,1,1, PROBLEM_SIZE + 1], issue_token=False, offset=0)
            bd_colidx = shim_dma_single_bd_task(of_in0a, inCombinedTensor,
                sizes=[1,1,1, NNZ_SIZE],           issue_token=False, offset=PROBLEM_SIZE + 1)
            bd_vals   = shim_dma_single_bd_task(of_in0a, inCombinedTensor,
                sizes=[1,1,1, NNZ_SIZE],           issue_token=True,  offset=PROBLEM_SIZE + 1 + NNZ_SIZE)


            in_X = shim_dma_single_bd_task(
                of_in1, inXTensor, sizes=[1, 1, 1, PROBLEM_SIZE], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out1, outTensor, sizes=[1, 1, 1, PROBLEM_SIZE], issue_token=True
            )

            dma_start_task(bd_rowptr, bd_colidx, bd_vals, in_X, out_task)
            dma_await_task(bd_vals, in_X, out_task)

# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    my_vector_bias_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

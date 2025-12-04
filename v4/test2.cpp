//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cstdint>

//*****************************************************************************
// Modify this section to customize buffer datatypes, initialization functions,
// and verify function. The other place to reconfigure your design is the
// Makefile.
//*****************************************************************************

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
// ------------------------------------------------------
// Configure this to match your buffer data type
// ------------------------------------------------------
#if INT_BIT_WIDTH == 16
using DATATYPE_IN1 = std::int16_t;
using DATATYPE_OUT = std::int16_t;
#else
using DATATYPE_IN1 = std::int32_t;
using DATATYPE_OUT = std::int32_t;
#endif
using DATATYPE_IN2 = std::int32_t;
#endif

// Initialize Input buffer 1
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE  ) {
    // [RBP = [rowptr (RPB+1) | colidx (CHUNK_NNZ) | vals (CHUNK_NNZ) ]]
    // para primera iteracion rbp = size, solo tengo una pagina
    int cols = 512;
    int nnz = 512;
    DATATYPE_IN1* rowptr = bufIn1;
    DATATYPE_IN1* colidx = bufIn1 + cols + 1;
    DATATYPE_IN1* vals   = bufIn1 + cols + 1 + nnz;
    for (int i = 0; i < cols; ++i) {
        rowptr[i] = i;
        colidx[i] = i;
        vals[i] = 1;
    }
    std::cout << "Data pirntes" << std::endl;

}

// Initialize Input buffer 2
void initialize_bufIn2(DATATYPE_IN2 *bufIn2, int SIZE) {
    for (int i = 0; i < 512; ++i) {
        bufIn2[i] = i;
    }   
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE);
}

// Functional correctness verifyer
int verify_vector_scalar_mul(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                             DATATYPE_OUT *bufOut, int SIZE, int verbosity) {
    int cols = 512;
    int nnz = 512;
    int errors = 0;
    DATATYPE_IN1* rowptr = bufIn1;
    DATATYPE_IN1* colidx = bufIn1 + cols + 1;
    DATATYPE_IN1* vals   = bufIn1 + cols + 1 + nnz;
    for (int i = 0; i < nnz; ++i) {
        int ref = colidx[i] * bufIn2[colidx[i]];
        int test = bufOut[i];
        if (test != ref) { 
            errors++;
        }
    }
    return errors;
}

//*****************************************************************************
// Should not need to modify below section
//*****************************************************************************

int main(int argc, const char *argv[]) {

  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(DATATYPE_IN1);
  constexpr int IN2_VOLUME = IN2_SIZE / sizeof(DATATYPE_IN2);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE_OUT);

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_IN2, DATATYPE_OUT,
                              initialize_bufIn1, initialize_bufIn2,
                              initialize_bufOut, verify_vector_scalar_mul>(
      IN1_VOLUME, IN2_VOLUME, OUT_VOLUME, myargs);
  return res;
}
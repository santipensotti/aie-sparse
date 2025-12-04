//===- smoke_spmv_id.cc -------------------------------------------*- C++ -*-===//
//
// Smoke test SpMV con A = Identidad (primer bloque de RPB filas).
// Layout esperado por el runtime_sequence actual:
//
//   combined_page = [ rowptr (RPB+1) | colidx (CHUNK_NNZ) | vals (CHUNK_NNZ) ]
//
// Con RPB=128, CHUNK_NNZ=1024 => COMBINED_PAGE_SIZE = (RPB+1) + 2*CHUNK_NNZ.
//
// sequence signature (algo así):
//   sequence(A_ALL: memref<COMBINED_PAGE_SIZExi32>,
//            X_VEC: memref<ROWSxi32>,
//            Y_VEC: memref<ROWSxi32>)
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//----------------------------------------------------------------------------

#include "cxxopts.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"

#include "test_utils.h"

// ---------- Parámetros del smoke ----------
static constexpr int RPB       = 128;   // rows per block (= filas que procesa este page)
static constexpr int CHUNK_NNZ = 1024;  // cota nnz por bloque
static constexpr int ROWS      = 512;   // tamaño total del vector X/Y
static constexpr int COLS      = 512;   // no lo usamos acá, pero matchea el diseño

// Combined = [ rowptr (RPB+1) | colidx (CHUNK_NNZ) | vals (CHUNK_NNZ) ]
static constexpr int NUM_PAGES = ROWS / RPB; // 512/128 = 4
static constexpr int COMBINED_PAGE_SIZE = (RPB + 1) + 2 * CHUNK_NNZ;
static constexpr int TOTAL_COMBINED_SIZE = NUM_PAGES * COMBINED_PAGE_SIZE;

// ---------- Helpers ----------
static inline void make_identity_all(std::vector<uint32_t>& combined_out) {
  combined_out.assign(TOTAL_COMBINED_SIZE, 0);

  for (int p = 0; p < NUM_PAGES; ++p) {
    uint32_t* base   = combined_out.data() + p * COMBINED_PAGE_SIZE;
    uint32_t* rowptr = base;
    uint32_t* colidx = base + (RPB + 1);
    uint32_t* vals   = base + (RPB + 1) + CHUNK_NNZ;

    for (int i = 0; i < RPB; ++i) {
      int global_row = p * RPB + i;
      if (global_row >= ROWS)
        break;

      // CSR local en la página: una nnz por fila
      rowptr[i] = i;              // fila i usa el índice i en colidx/vals
      colidx[i] = global_row;     // identidad: columna = fila global
      vals[i]   = 1U;
    }
    rowptr[RPB] = RPB;            // total nnz usadas en esta página
  }
}


static inline int verify_identity(const uint32_t *x, const uint32_t *y) {
  int errors = 0;
  for (int i = 0; i < RPB; ++i) {
    if (y[i] != x[i]) {
      std::cerr << "Mismatch fila " << i << " ref=" << x[i] << " got=" << y[i] << "\n";
      ++errors;
    }
  }
  return errors;
}

int main(int argc, const char *argv[]) {
  // ----------------- CLI -----------------
  cxxopts::Options options("smoke_spmv_id");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;

  test_utils::parse_options(argc, argv, options, vm);

  const int verbosity = vm["verbosity"].as<int>();
  const bool do_verify = vm["verify"].as<bool>();

  // ----------------- XRT setup -----------------
  unsigned device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());

  const std::string Node = vm["kernel"].as<std::string>();
  if (verbosity) std::cout << "Kernel opcode prefix: " << Node << "\n";

  auto xkernels = xclbin.get_kernels();
  auto it = std::find_if(xkernels.begin(), xkernels.end(),
                         [&](xrt::xclbin::kernel &k) {
                           auto name = k.get_name();
                           if (verbosity) std::cout << "Found kernel symbol: " << name << "\n";
                           return name.rfind(Node, 0) == 0;
                         });
  if (it == xkernels.end()) {
    std::cerr << "Kernel '" << Node << "' not found in xclbin.\n";
    return 2;
  }
  auto kernelName = it->get_name();

  if (verbosity)
    std::cout << "Loading instr ELF: " << vm["instr"].as<std::string>() << "\n";
  xrt::elf elf(vm["instr"].as<std::string>());
  xrt::module mod{elf};

  if (verbosity) std::cout << "Creating kernel handle: " << kernelName << "\n";
  auto kernel = xrt::ext::kernel(context, mod, kernelName);

  // ----------------- Buffers host -----------------
  // sequence signature actual: (A_ALL, X_VEC, Y_VEC)
  //
  // Usamos xrt::ext::bo enlazados al kernel y al contexto, uno por cada argumento.
  xrt::bo bo_A_all = xrt::ext::bo{device, TOTAL_COMBINED_SIZE * sizeof(uint32_t)};
  xrt::bo bo_x     = xrt::ext::bo{device, ROWS * sizeof(uint32_t)};
  xrt::bo bo_y     = xrt::ext::bo{device, ROWS * sizeof(uint32_t)};

  auto *h_A = bo_A_all.map<uint32_t *>();
  auto *h_x = bo_x.map<uint32_t *>();
  auto *h_y = bo_y.map<uint32_t *>();

  // ----- Construir COMBINED_PAGE (identidad en el bloque) -----
    std::vector<uint32_t> all_pages;
    make_identity_all(all_pages);
    std::memcpy(h_A, all_pages.data(), all_pages.size() * sizeof(uint32_t));

  // ----- X sencillo -----
  for (int i = 0; i < ROWS; ++i)
    h_x[i] = (i + 1) * 10;   //  [10,20,...]

  std::fill(h_y, h_y + ROWS, 0u);

  bo_A_all.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ----------------- Lanzar kernel -----------------
  if (verbosity) std::cout << "Running kernel…\n";
  const unsigned opcode = 0; // el microcódigo (insts.elf) asocia opcode=0 a sequence()
  auto run = kernel(opcode, 0, 0, bo_A_all, bo_x, bo_y);
  run.wait();

  bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // ----------------- Verificar -----------------
  int errors = 0;
  if (do_verify) {
    errors = verify_identity(h_x, h_y);
  }

  if (!errors) {
    std::cout << "PASS\n";
    return 0;
  } else {
    std::cout << "FAIL (" << errors << " errores)\n";
    std::cout << "Y_OUT (primeras " << RPB << " filas): ";
    for (int i = 0; i < RPB; ++i)
      std::cout << h_y[i] << (i + 1 < RPB ? ' ' : '\n');

    std::cout << "X_VEC (primeras " << RPB << " filas): ";
    for (int i = 0; i < RPB; ++i)
      std::cout << h_x[i] << (i + 1 < RPB ? ' ' : '\n');

    return 1;
  }
}

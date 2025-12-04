#include "cxxopts.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>
#include <numeric>   
#include <cstring>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"

#include "test_utils.h"

// --- generator: todo int32_t ---
size_t fill_random_csr_arrays_simple(int rows, int cols,
                                     uint32_t* rowptr_out,
                                     uint32_t* colidx_out,
                                     uint32_t* vals_out,
                                     int nnz_target_row = 3,
                                     bool jitter_poisson = false,
                                     uint64_t seed = 42,
                                     int vmin = 1, int vmax = 9) {
  if (rows <= 0 || cols <= 0) { rowptr_out[0] = 0; return 0; }
  if (nnz_target_row < 0) nnz_target_row = 0;
  if (nnz_target_row > cols) nnz_target_row = cols;

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> valdist(vmin, vmax);

  std::vector<int> all_cols(cols);
  std::iota(all_cols.begin(), all_cols.end(), 0);

  size_t nnz = 0;
  rowptr_out[0] = 0;

  for (int r = 0; r < rows; ++r) {
    int k = jitter_poisson ? std::poisson_distribution<int>(nnz_target_row)(rng)
                           : nnz_target_row;
    if (k < 0) k = 0; if (k > cols) k = cols;

    std::shuffle(all_cols.begin(), all_cols.end(), rng);
    std::sort(all_cols.begin(), all_cols.begin() + k);

    for (int i = 0; i < k; ++i) {
      colidx_out[nnz + i] = static_cast<uint32_t>(all_cols[i]);
      vals_out  [nnz + i] = static_cast<uint32_t>(valdist(rng));
    }
    nnz += static_cast<size_t>(k);
    rowptr_out[r + 1] = static_cast<uint32_t>(nnz);
  }
  return nnz;
}

// --- checker CSR int32_t ---
int check_csr(const uint32_t* rowptr, const uint32_t* colidx, const uint32_t* vals,
              const uint32_t*  x, const uint32_t* y_out, int rows) {
  int errors = 0;
  for (int i = 0; i < rows; ++i) {
    const int32_t begin = rowptr[i];
    const int32_t end   = rowptr[i + 1];
    long long acc = 0;
    for (int32_t k = begin; k < end; ++k) {
      acc += static_cast<long long>(vals[k]) * static_cast<long long>(x[colidx[k]]);
    }
    const int32_t ref = static_cast<int32_t>(acc);
    if (y_out[i] != ref) {
      ++errors;
      std::cout << "Error fila " << i << " Esperaba " << ref << " obtuve: " << y_out[i] << std::endl;
    }
  }
  return errors;
}


int main(int argc, const char *argv[]) {
  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("Vector Scalar Add Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);

  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();


constexpr int PROBLEM_SIZE  = 512;
constexpr int NNZ           = 1536;
constexpr int NNZ_ROW           = 3;
constexpr int ROWPTR_SIZE   = PROBLEM_SIZE + 1;
constexpr int COMBINED_SIZE = ROWPTR_SIZE + NNZ * 2;
constexpr int IN_SIZE       = PROBLEM_SIZE;
constexpr int OUT_SIZE      = PROBLEM_SIZE;


  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Load the kernel
  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  // Register xclbin
  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";
  device.register_xclbin(xclbin);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // Load instr ELF
  xrt::elf elf(vm["instr"].as<std::string>());
  xrt::module mod{elf};

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::ext::kernel(context, mod, kernelName);

  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------

  xrt::bo bo_inCombined = xrt::ext::bo{device, COMBINED_SIZE * sizeof(int32_t)};
  xrt::bo bo_inX = xrt::ext::bo{device, IN_SIZE * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, OUT_SIZE * sizeof(int32_t)};

  std::cout << "Writing data into buffer objects.\n";
  auto *combined = bo_inCombined.map<uint32_t *>();
    uint32_t *rowptr = combined + 0;
    uint32_t *colidx = combined + ROWPTR_SIZE;     
    uint32_t *vals   = combined + ROWPTR_SIZE + NNZ; 

  
  size_t nnz = fill_random_csr_arrays_simple(PROBLEM_SIZE, PROBLEM_SIZE,
                                           rowptr, colidx, vals,
                                           NNZ_ROW,
                                           /*jitter_poisson=*/false ,
                                           /*seed=*/123);

  uint32_t *bufInX = bo_inX.map<uint32_t *>();
  std::vector<uint32_t> srcVecX;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecX.push_back(i + 1);
  memcpy(bufInX, srcVecX.data(), (srcVecX.size() * sizeof(uint32_t)));


  bo_inCombined.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inX.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    

  if (verbosity >= 1)
  std::cout << "Running Kernel.\n";
  // Print the values of rowptr, colidx, vals
  std::cout << "rowptr: ";
  for (int i = 0; i < ROWPTR_SIZE; i++) {
    std::cout << rowptr[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "colidx: ";
  for (int i = 0; i < nnz; i++) {
    std::cout << colidx[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "vals: ";
  for (int i = 0; i < nnz; i++) {
    std::cout << vals[i] << " ";
  }
  std::cout << std::endl;
    
  std::cout << "x: ";
  for (int i = 0; i < IN_SIZE; i++) {
    std::cout << srcVecX[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Original matrix" << std::endl;
  for (int i = 0; i < PROBLEM_SIZE; i++) {
    const int32_t begin = rowptr[i];
    const int32_t end   = rowptr[i + 1];
    std::cout << "Row " << i << ": ";
    for (int32_t k = begin; k < end; ++k) {
      std::cout << "(" << colidx[k] << ", " << vals[k] << ") ";
    }
    std::cout << std::endl;
  }

  unsigned int opcode = 3;
  auto run = kernel(opcode, 0, 0, bo_inCombined, bo_inX, bo_out);
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut = bo_out.map<uint32_t *>();

  int errors = check_csr(rowptr, colidx, vals, srcVecX.data(), bufOut, PROBLEM_SIZE);

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    std::cout << errors << std::endl;
    return 1;
  }
}

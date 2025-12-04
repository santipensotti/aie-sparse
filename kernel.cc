#include <aie_api/aie.hpp>
#include "aie_api/utils.hpp"
#include <cstdint>
#include <cstdio>
extern "C" {
void spmv_sparse(const int32_t* __restrict combined,
                 const int32_t* __restrict x,
                 int32_t* __restrict y) {
  constexpr int N = 512;                 // <-- Debe MATCHEAR PROBLEM_SIZE del host

  // Layout: [ rowptr (N+1) | colidx (NNZ) | vals (NNZ) ]
  const int32_t* rowptr = combined;           // N+1
  const int      NNZ    = rowptr[N];          // CSR: última entrada es nnz total
  const int32_t* colidx = rowptr + (N + 1);   // salteo N+1
  const int32_t* vals   = colidx + NNZ;       // salteo NNZ
  printf("tesinteand");
  for (int i = 0; i < N; ++i) {
    const int32_t begin = rowptr[i];
    const int32_t end   = rowptr[i + 1];
    int32_t sum = 0;
    for (int32_t k = begin; k < end; ++k)
      sum += vals[k] * x[colidx[k]];
    y[i] = sum;
  }
}
}

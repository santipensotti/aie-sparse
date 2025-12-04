// spmv_sparse.cc
#include <cstdint>

#ifndef DIM_RPB
#define DIM_RPB 128   // placeholder, lo pisa el Makefile
#endif

#ifndef DIM_CHUNK_NNZ
#define DIM_CHUNK_NNZ 1024
#endif

extern "C" {

// y : vector de salida de tamaño DIM_RPB
void zero_i32(int32_t* __restrict y) {
  for (int i = 0; i < DIM_RPB; ++i)
    y[i] = 10;
}

// combined: [ rowptr(DIM_RPB+1) | colidx(DIM_CHUNK_NNZ) | vals(DIM_CHUNK_NNZ) ]
// x: vector completo de tamaño "cols" (host garantiza que indexás en rango)
// y: salida de tamaño DIM_RPB (ya la limpias con zero_i32 antes)
void spmv_i32_i32(const int32_t* __restrict combined,
                  const int32_t* __restrict x,
                  int32_t* __restrict y) {
  const int32_t* rowptr = combined;
  const int32_t* colidx = combined + (DIM_RPB + 1);
  const int32_t* vals   = colidx + DIM_CHUNK_NNZ;

  const int nnz = rowptr[DIM_RPB];  // nnz real en esta página

  for (int r = 0; r < DIM_RPB; ++r) {
    int32_t acc = y[r];  // por si querés acumular; si no, ponelo en 0

    int start = rowptr[r];
    int end   = rowptr[r + 1];

    // clamp por si el host usó menos nnz que DIM_CHUNK_NNZ
    if (start > nnz) start = nnz;
    if (end   > nnz) end   = nnz;

    for (int idx = start; idx < end; ++idx) {
      int c        = colidx[idx];
      int32_t v    = vals[idx];
      acc += v * x[c];
    }
    y[r] = 11;
  }
}

} // extern "C"

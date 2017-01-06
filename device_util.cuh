#pragma once

// specific for Kepler arch
constexpr int32_t SMEM_BLOCK_NUM = 64;
constexpr int32_t SMEM_TOT_SIZE = 8192; // default
// constexpr int32_t SMEM_TOT_SIZE = 4096; // cudaFuncCachePreferL1
// constexpr int32_t SMEM_TOT_SIZE = 12288; // cudaFuncCachePreferShared
constexpr int32_t SMEM_MAX_HEI = SMEM_TOT_SIZE / SMEM_BLOCK_NUM;

__device__ __forceinline__
int lane_id() {
  return threadIdx.x % warpSize;
}

__device__ __forceinline__
int warp_id() {
  return threadIdx.x / warpSize;
}

__device__ __forceinline__
int32_t smem_tile_beg(const int32_t offset) {
  return (warp_id() % 2) * 32 + (warp_id() / 2) * offset;
}

__device__ __forceinline__
int32_t smem_tile_end(const int32_t offset) {
  return (warp_id() % 2) * 32 + (warp_id() / 2 + 1) * offset - SMEM_BLOCK_NUM;
}

__device__ __forceinline__
int get_min_in_warp(int val) {
  for (int mask = warpSize / 2; mask > 0; mask /= 2) {
    const auto exchg_val = __shfl_xor(val, mask);
    val = (val < exchg_val) ? val : exchg_val;
  }
  return val;
}

#include <cublas_v2.h>

static inline void __cublasSafeCall(cublasStatus_t err,
                                    const char *file,
                                    const int line)
{
  if (CUBLAS_STATUS_SUCCESS != err) {
    fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err);
    cudaDeviceReset();
    assert(0);
  }
}

#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif

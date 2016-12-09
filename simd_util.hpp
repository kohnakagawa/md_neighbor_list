#pragma once

#include <immintrin.h>

typedef double v4df __attribute__((vector_size(32)));

static inline void transpose_4x4(v4df& row0,
                                 v4df& row1,
                                 v4df& row2,
                                 v4df& row3) {
  v4df tmp0, tmp1, tmp2, tmp3;
  tmp0 = _mm256_unpacklo_pd(row0, row1);
  tmp1 = _mm256_unpackhi_pd(row0, row1);
  tmp2 = _mm256_unpacklo_pd(row2, row3);
  tmp3 = _mm256_unpackhi_pd(row2, row3);

  row0 = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
  row1 = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
  row2 = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);
  row3 = _mm256_permute2f128_pd(tmp1, tmp3, 0x31);
}

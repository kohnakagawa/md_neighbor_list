#pragma once

#include <x86intrin.h>
#include <iostream>

typedef double v4df __attribute__((vector_size(32)));
typedef double v8df __attribute__((vector_size(64)));
typedef int32_t v8si __attribute__((vector_size(32)));
typedef int32_t v16si __attribute__((vector_size(64)));
typedef int64_t v4di __attribute__((vector_size(32)));
typedef int64_t v8di __attribute__((vector_size(64)));

static inline v8df _mm512_load2_m256d(const double* hiaddr,
                                      const double* loaddr) {
  v8df ret = _mm512_castpd256_pd512(_mm256_load_pd(loaddr));
  ret = _mm512_insertf64x4(ret, _mm256_load_pd(hiaddr), 0x1);
  return ret;
}

static inline void _mm512_store2_m256d(double* hiaddr,
                                       double* loaddr,
                                       const v8df& dat) {
  _mm256_store_pd(loaddr, _mm512_castpd512_pd256(dat));
  _mm256_store_pd(hiaddr, _mm512_extractf64x4_pd(dat, 0x1));
}

static inline void transpose_4x4x2(v8df& va,
                                   v8df& vb,
                                   v8df& vc,
                                   v8df& vd) {
  v8df t_a = _mm512_unpacklo_pd(va, vb);
  v8df t_b = _mm512_unpackhi_pd(va, vb);
  v8df t_c = _mm512_unpacklo_pd(vc, vd);
  v8df t_d = _mm512_unpackhi_pd(vc, vd);

  va = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_c);
  vb = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_d);
  vc = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_c);
  vd = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_d);
}

static inline void transpose_4x4x2(const v8df& va,
                                   const v8df& vb,
                                   const v8df& vc,
                                   const v8df& vd,
                                   v8df& vx,
                                   v8df& vy,
                                   v8df& vz) {
  v8df t_a = _mm512_unpacklo_pd(va, vb);
  v8df t_b = _mm512_unpackhi_pd(va, vb);
  v8df t_c = _mm512_unpacklo_pd(vc, vd);
  v8df t_d = _mm512_unpackhi_pd(vc, vd);

  vx = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_c);
  vy = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_d);
  vz = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_c);
}

static inline void transpose_4x4(const v4df& va,
                                 const v4df& vb,
                                 const v4df& vc,
                                 const v4df& vd,
                                 v4df& vx,
                                 v4df& vy,
                                 v4df& vz) {
  v4df tmp0 = _mm256_unpacklo_pd(va, vb);
  v4df tmp1 = _mm256_unpackhi_pd(va, vb);
  v4df tmp2 = _mm256_unpacklo_pd(vc, vd);
  v4df tmp3 = _mm256_unpackhi_pd(vc, vd);
  vx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
  vy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
  vz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);
}

static inline v8df _mm512_rot_rshift_b64(const v8df& a,
                                         const int shift) {
  return _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(a),
                                                 _mm512_castpd_si512(a),
                                                 shift));
}

static inline v8di _mm512_rot_rshift_b64(const v8di& a,
                                         const int shift) {
  return _mm512_alignr_epi64(a, a, shift);
}

static inline void print512(v8df r) {
  union {
    v8df r;
    double elem[8];
  } tmp;
  tmp.r = r;
  printf("%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
         tmp.elem[0], tmp.elem[1], tmp.elem[2], tmp.elem[3],
         tmp.elem[4], tmp.elem[5], tmp.elem[6], tmp.elem[7]);
}

#pragma once

#include <immintrin.h>

typedef double v4df __attribute__((vector_size(32)));
typedef int32_t v8si __attribute__((vector_size(32)));
typedef int64_t v4di __attribute__((vector_size(32)));

const int32_t shfl_table_[16][8] = {
  {0, 0, 0, 0, 0, 0, 0, 0},
  {6, 7, 0, 0, 0, 0, 0, 0},
  {4, 5, 0, 0, 0, 0, 0, 0},
  {4, 5, 6, 7, 0, 0, 0, 0},
  {2, 3, 0, 0, 0, 0, 0, 0},
  {2, 3, 6, 7, 0, 0, 0, 0},
  {2, 3, 4, 5, 0, 0, 0, 0},
  {2, 3, 4, 5, 6, 7, 0, 0},
  {0, 1, 0, 0, 0, 0, 0, 0},
  {0, 1, 6, 7, 0, 0, 0, 0},
  {0, 1, 4, 5, 0, 0, 0, 0},
  {0, 1, 4, 5, 6, 7, 0, 0},
  {0, 1, 2, 3, 0, 0, 0, 0},
  {0, 1, 2, 3, 6, 7, 0, 0},
  {0, 1, 2, 3, 4, 5, 0, 0},
  {0, 1, 2, 3, 4, 5, 6, 7}
};

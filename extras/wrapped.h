#pragma once
#include "ruy/platform.h"

#include <cstdint>

#define MOZINTGEMM_BATTERIES_INCLUDED

#if 0
#include "matrix.h"
#define PRINT_MATRIX_DEBUG(d, rows, cols, order)                               \
  do {                                                                         \
    if (std::getenv("ARM_PLAYGROUND_DEBUG")) {                                 \
      std::cout << #d << ": " << std::endl;                                    \
      pg::utils::printMatrix(std::cout, d, pg::Layout(rows, cols, order));     \
    }                                                                          \
  } while (0)

#endif

#if RUY_PLATFORM_X86
#include "3rd-party/intgemm/intgemm/intgemm.h"
#include "moz_intgemm.h"
#include <cstdint>
#include <iostream>
#endif

#include "ruy/ruy.h"
#include "ruy/system_aligned_alloc.h"
#include <algorithm>
#include <cassert>
#include <cmath>

#if RUY_PLATFORM_NEON
#include <arm_neon.h>
#endif

using Index = uint32_t;

namespace pg::Ruy {

#include "MozIntGemm/detail.inl"
#include "MozIntGemm/moz_intgemm.inl"

namespace detail {

template <class Path> struct Preprocess;
struct kStandardCpp;
struct kNeon;

} // namespace detail

} // namespace pg::Ruy

#if RUY_PLATFORM_X86

namespace pg::Intgemm {

#include "MozIntGemm/moz_intgemm.inl"
} // namespace pg::Intgemm

#endif

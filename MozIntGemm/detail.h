#pragma once
#include "moz_intgemm.h"
#include "ruy/platform.h"
#include "ruy/system_aligned_alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

#if RUY_PLATFORM_NEON
#include <arm_neon.h>
#endif

#include "detail.inl"

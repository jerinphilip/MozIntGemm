#include "detail.h"
#include "firefox_interface.h"
#include "ruy/ruy.h"
#include "ruy/system_aligned_alloc.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

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

namespace pg::Ruy {
// The following is bad practice. But this makes it easy for @jerinphilip to
// export the cpp file directly for Mozilla as an implementation.
#define RUY_BATTERIES_ALREADY_INCLUDED
#include "impl_ruy-export.cpp"

} // namespace pg::Ruy

#include "matrix.h"
#include "wrapped.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>

#define DEBUG_PRINTABLE(x)                                                     \
  do {                                                                         \
    if (std::getenv("ARM_PLAYGROUND_DEBUG")) {                                 \
      std::cout << #x << ":\n" << x << std::endl;                              \
    }                                                                          \
  } while (0)

namespace {
using namespace pg::Ruy::detail;
using namespace pg;

template <class Path>
void Quantize(Matrix<float> &input, Matrix<int8_t> &output) {
  Preprocess<Path>::quantize(input.data(), 127.0f, 0, input.nrows(),
                             input.ncols(), output.data());
}

TEST(PreprocOnARM, NeonVsStandard) {
  std::mt19937_64 gen64;
  const size_t M = 8, N = 64, P = 64;
  auto [A, B, bias] = generateInput(gen64, M, N, P);
  Matrix<int8_t> quantizedAStd(A.layout()), quantizedANeon(A.layout());

  Quantize<kStandardCpp>(A, quantizedAStd);
  Quantize<kNeon>(A, quantizedANeon);
  DEBUG_PRINTABLE(quantizedAStd);
  DEBUG_PRINTABLE(quantizedANeon);

  const float MSE_TOLERANCE = 1e-9;
  auto mse = MeanSquaredError(quantizedAStd, quantizedANeon);
  ASSERT_LT(mse, MSE_TOLERANCE);
  DEBUG_PRINTABLE(mse);
}

} // namespace

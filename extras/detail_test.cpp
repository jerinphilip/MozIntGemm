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

TEST(PreprocOnARM, QuantizeNeonVsStandard) {
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

template <class Path>
void Transpose(Matrix<int8_t> &input, Matrix<int8_t> &output) {
  Preprocess<Path>::transpose(input.data(), input.nrows(), input.ncols(),
                              output.data());
}

TEST(PreprocOnARM, TransposeNeonVsStandard) {
  std::mt19937_64 gen64;
  gen64.seed(42);

  const size_t M = 8, N = 64, P = 64;
  Layout layout(M, N, Order::RowMajor);
  auto A = make_random_matrix<int8_t>(gen64, layout, -127, 127);
  Matrix<int8_t> transposedAStd(A.layout().transpose()),
      transposedANeon(A.layout().transpose());

  Transpose<kStandardCpp>(A, transposedAStd);
  Transpose<kNeon>(A, transposedANeon);
  DEBUG_PRINTABLE(transposedAStd);
  DEBUG_PRINTABLE(transposedANeon);

  const float MSE_TOLERANCE = 1e-9;
  auto mse = MeanSquaredError(transposedAStd, transposedANeon);
  ASSERT_LT(mse, MSE_TOLERANCE);
  DEBUG_PRINTABLE(mse);
}

} // namespace

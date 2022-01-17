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

template <class Path>
void UnQuantizeAddBias(Matrix<int32_t> &intermediate, Matrix<float> &bias,
                       Matrix<float> &output) {
  float unquant_multiplier = 1 / 127.0f;
  Preprocess<Path>::unquantizeAddBias(intermediate.data(), bias.data(),
                                      unquant_multiplier, intermediate.nrows(),
                                      intermediate.ncols(), output.data());
}

TEST(PreprocOnARM, UnquantizeAddBiasNeonVsStandard) {
  std::mt19937_64 gen64;
  const size_t M = 8, N = 64, P = 64;
  auto [A, B, bias] = generateInput(gen64, M, N, P);
  Layout productLayout(A.nrows(), B.ncols(), Order::RowMajor);
  auto intermediate =
      make_random_matrix<int32_t>(gen64, productLayout, -127, 127);

  Matrix<float> outputStd(productLayout), outputNeon(productLayout);

  UnQuantizeAddBias<kStandardCpp>(intermediate, bias, outputStd);
  UnQuantizeAddBias<kNeon>(intermediate, bias, outputNeon);

  DEBUG_PRINTABLE(outputStd);
  DEBUG_PRINTABLE(outputNeon);

  const float MSE_TOLERANCE = 1e-9;
  auto mse = MeanSquaredError(outputStd, outputNeon);
  ASSERT_LT(mse, MSE_TOLERANCE);
  DEBUG_PRINTABLE(mse);
}

TEST(PreprocOnARM, TransposeDriver) {
  constexpr size_t tile = 16;
  constexpr size_t block = tile * tile;
  std::vector<int16_t> src(block), dst(block);
  std::iota(src.begin(), src.end(), 0);
  std::fill(dst.begin(), dst.end(), 0);
  Preprocess<kNeon>::transpose_16x16(src.data(), dst.data());
  auto print = [tile, block](const std::vector<auto> &m) {
    for (size_t i = 0; i < tile; i++) {
      for (size_t j = 0; j < tile; j++) {
        if (j != 0) {
          std::cout << " ";
        }
        auto v = m[i * tile + j];
        std::cout << std::setfill('0') << std::setw(3) << v;
      }
      std::cout << "\n";
    }
  };

  print(src);
  print(dst);
}

} // namespace

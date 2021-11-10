#include "3rd-party/intgemm/intgemm/aligned.h"
#include "firefox_interface.h"
#include "matrix.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <iostream>
#include <random>

namespace {

template <class T>
std::ostream &operator<<(std::ostream &out,
                         const intgemm::AlignedVector<T> &v) {
  for (size_t i = 0; i < v.size(); i++) {
    if (i != 0)
      out << " ";
    const T *data = v.begin();
    std::cout << (int)data[i];
  }
  return out;
}

using namespace pg;

// Repeats path for lib with matrices A, B and bias. Final result goes into
// output.
#define REPEAT_PATH(lib, A, B, bias, output, output_scale)                     \
  std::cout << #A << A;                                                        \
  std::cout << #B << B;                                                        \
  intgemm::AlignedVector<int8_t> mA_prepared(A.nrows() * A.ncols()),           \
      mB_prepared(B.nrows() * B.ncols());                                      \
  intgemm::AlignedVector<float> mBias_prepared(bias.nrows() * bias.ncols());   \
                                                                               \
  int8_t *A_prepared = mA_prepared.begin();                                    \
  int8_t *B_prepared = mB_prepared.begin();                                    \
  float *bias_prepared = mBias_prepared.begin();                               \
                                                                               \
  lib::int8PrepareA(A.data(), A.scale(), A.zero_point(), A.nrows(), A.ncols(), \
                    A_prepared);                                               \
                                                                               \
  std::cout << "A_prepared: " << mA_prepared << "\n";                          \
  lib::int8PrepareB(B.data(), B.scale(), B.zero_point(), B.nrows(), B.ncols(), \
                    B_prepared);                                               \
                                                                               \
  std::cout << "B_prepared: " << mB_prepared << "\n";                          \
  lib::int8PrepareBias(B_prepared, A.scale(), A.zero_point(), B.scale(),       \
                       B.zero_point(), B.nrows(), B.ncols(), bias.data(),      \
                       bias_prepared);                                         \
                                                                               \
  std::cout << "bias_prepared: " << mBias_prepared << "\n";                    \
  lib::int8MultiplyAndAddBias(A_prepared, A.scale(), A.zero_point(),           \
                              B_prepared, B.scale(), B.zero_point(),           \
                              bias_prepared, output_scale, A.nrows(),          \
                              A.ncols(), B.ncols(), output);

TEST(EndToEnd, EndToEnd) {
  std::mt19937_64 gen64;
  gen64.seed(42);
  constexpr size_t DIM_MAX = 32;
  constexpr size_t DIM_MIN = 16;
  constexpr size_t MC_RUNS = 1;
  for (size_t i = 0; i < MC_RUNS; i++) {
    std::uniform_int_distribution<> distribution(DIM_MIN, DIM_MAX);

    size_t M, N, P; // A = M x N matrix, B = N x P

    M = distribution(gen64);
    N = distribution(gen64);
    P = distribution(gen64);

    // Do some stuff to get stuff rounded to multiples of 8
    const size_t _WIDTH = 64;
    M = ((M / _WIDTH) + 1) * _WIDTH;
    N = ((N / _WIDTH) + 1) * _WIDTH;
    P = ((P / _WIDTH) + 1) * _WIDTH;
    M = 1, N = 16, P = 8;
    // M = 32, N = 32, P = 32;

    std::cout << "Dimensions: A[" << M << "x" << N << "];  B[" << N << "x" << P
              << "]\n\n";
    Matrix<float> B(N, P), A(M, N);
    A.fill(gen64);
    B.fill(gen64, -8, 8);
    Matrix<float> bias(1, P);
    bias.fill(gen64);

    float output_scale = 1.0f;
    Matrix<float> intgemmProduct(A.nrows(), B.ncols());
    { REPEAT_PATH(Intgemm, A, B, bias, intgemmProduct.data(), output_scale); }
    Matrix<float> ruyProduct(A.nrows(), B.ncols());
    { REPEAT_PATH(Ruy, A, B, bias, ruyProduct.data(), output_scale); }

    float mse = MeanSquaredError(ruyProduct, intgemmProduct);
    std::cout << "ruyProduct: " << ruyProduct;
    std::cout << "intgemmProduct: " << intgemmProduct;
    std::cout << "Mean-Squared-Error(ruyProduct, intgemmProduct) = " << mse
              << "\n";
    ASSERT_NEAR(mse, 0.0f, /*abs_error=*/1e-7);
  }
}

} // namespace

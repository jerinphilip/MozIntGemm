#include "3rd-party/intgemm/intgemm/aligned.h"
#include "firefox_interface.h"
#include "matrix.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

namespace {

const float FMIN = -127.0f;
const float FMAX = 127.0f;

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

template <class Scalar, class AccumScalar>
Matrix<AccumScalar> ReferenceMultiply(const Matrix<Scalar> &A,
                                      const Matrix<Scalar> &B,
                                      const Matrix<Scalar> &bias) {

  Layout productLayout(A.nrows(), B.ncols(), Order::RowMajor);
  Matrix<AccumScalar> product(productLayout);
  std::fill(product.begin(), product.end(), 0);
  for (size_t i = 0; i < A.nrows(); i++) {
    for (size_t j = 0; j < B.ncols(); j++) {
      for (size_t k = 0; k < B.nrows(); k++) {
        product.at(i, j) += A.at(i, k) * B.at(k, j);
      }
      product.at(i, j) += bias.at(0, j);
    }
  }
  return product;
}

#define DEBUG_PRINTABLE(x)                                                     \
  do {                                                                         \
    if (std::getenv("ARM_PLAYGROUND_DEBUG")) {                                 \
      std::cout << #x << ": " << x << std::endl;                               \
    }                                                                          \
  } while (0)

#define forwardCallToNamespace(ns, fn)                                         \
  template <class... Args> static void fn(Args... args) {                      \
    /*std::cerr << "Calling " << #ns << "::" << #fn << std::endl;*/            \
    ns::fn(args...);                                                           \
  }

#define namespaceToStructForTemplating(ns)                                     \
  struct _##ns {                                                               \
    forwardCallToNamespace(ns, int8PrepareA);                                  \
    forwardCallToNamespace(ns, int8PrepareB);                                  \
    forwardCallToNamespace(ns, int8PrepareBias);                               \
    forwardCallToNamespace(ns, int8MultiplyAndAddBias);                        \
    forwardCallToNamespace(ns, int8SelectColumnsOfB);                          \
    forwardCallToNamespace(ns, int8PrepareBFromQuantizedTransposed);           \
  }

namespaceToStructForTemplating(Intgemm);
namespaceToStructForTemplating(Ruy);

void run(std::mt19937_64 &gen64,
         std::function<void(size_t, size_t, size_t)> f) {
  constexpr size_t DIM_MAX = 128;
  constexpr size_t DIM_MIN = 64;
  constexpr size_t MC_RUNS = 10;
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
    // M = 1, N = 16, P = 8;
    f(M, N, P);
  }
}

// Repeats path for lib with matrices A, B and bias. Final result goes into
// output, applied with an optional scale.
template <class Lib>
void MulABAddBias(Matrix<float> &A, Matrix<float> &B, Matrix<float> &bias,
                  float *output, float output_scale) {
  intgemm::AlignedVector<int8_t> mA_prepared(A.layout().num_elem()),
      mB_prepared(B.layout().num_elem());
  intgemm::AlignedVector<float> mBias_prepared(bias.layout().num_elem());

  int8_t *A_prepared = mA_prepared.begin();
  int8_t *B_prepared = mB_prepared.begin();
  float *bias_prepared = mBias_prepared.begin();

  Lib::int8PrepareA(A.data(), A.scale(), A.zero_point(), A.nrows(), A.ncols(),
                    A_prepared);

  Lib::int8PrepareB(B.data(), B.scale(), B.zero_point(), B.nrows(), B.ncols(),
                    B_prepared);

  Lib::int8PrepareBias(B_prepared, A.scale(), A.zero_point(), B.scale(),
                       B.zero_point(), B.nrows(), B.ncols(), bias.data(),
                       bias_prepared);

  Lib::int8MultiplyAndAddBias(A_prepared, A.scale(), A.zero_point(), B_prepared,
                              B.scale(), B.zero_point(), bias_prepared,
                              output_scale, A.nrows(), A.ncols(), B.ncols(),
                              output);
}

TEST(IntgemmVsRuy, NaiveMultiply) {
  std::mt19937_64 gen64;
  gen64.seed(42);
  auto f = [&gen64](size_t M, size_t N, size_t P) {
    Layout a_layout(M, N, Order::RowMajor);
    Layout b_layout(N, P, Order::RowMajor);
    Layout bias_layout(1, P, Order::RowMajor);

    auto A = make_random_matrix_but_int_values(gen64, a_layout, FMIN, FMAX);
    auto B = make_random_matrix_but_int_values(gen64, b_layout, FMIN, FMAX);
    auto bias =
        make_random_matrix_but_int_values(gen64, bias_layout, FMIN, FMAX);

    float output_scale = 1.0f;
    Layout productLayout(M, P, Order::RowMajor);
    Matrix<float> intgemmProduct(productLayout);
    MulABAddBias<_Intgemm>(A, B, bias, intgemmProduct.data(), output_scale);
    Matrix<float> ruyProduct(productLayout);
    MulABAddBias<_Ruy>(A, B, bias, ruyProduct.data(), output_scale);

    Matrix<float> refMul = ReferenceMultiply<float, float>(A, B, bias);
    float ruy_mse = MeanSquaredError(ruyProduct, refMul);
    float intgemm_mse = MeanSquaredError(intgemmProduct, refMul);
    DEBUG_PRINTABLE(A);
    DEBUG_PRINTABLE(B);
    DEBUG_PRINTABLE(bias);
    DEBUG_PRINTABLE(refMul);
    DEBUG_PRINTABLE(ruyProduct);
    DEBUG_PRINTABLE(intgemmProduct);
    ASSERT_NEAR(intgemm_mse, 0.0f, /*abs_error=*/1e-7);
    ASSERT_NEAR(ruy_mse, 0.0f, /*abs_error=*/1e-7);
  };
  run(gen64, f);
}

template <class Lib>
void MulASelectBAddBias(Matrix<float> &A, Matrix<float> &B, Matrix<float> &bias,
                        Index *cols_begin, Index num_cols, float *output,
                        float output_scale) {
  intgemm::AlignedVector<int8_t> mA_prepared(A.layout().num_elem()),
      mB_prepared(B.layout().num_elem());
  intgemm::AlignedVector<float> mBias_prepared(bias.layout().num_elem());

  int8_t *A_prepared = mA_prepared.begin();
  int8_t *B_prepared = mB_prepared.begin();
  float *bias_prepared = mBias_prepared.begin();

  Layout selected_b_layout(B.nrows(), num_cols, Order::ColMajor);
  intgemm::AlignedVector<int8_t> mBSelected(selected_b_layout.num_elem());
  int8_t *B_prepared_selected = mBSelected.begin();

  Lib::int8PrepareA(A.data(), A.scale(), A.zero_point(), A.nrows(), A.ncols(),
                    A_prepared);

  Lib::int8PrepareB(B.data(), B.scale(), B.zero_point(), B.nrows(), B.ncols(),
                    B_prepared);

  Lib::int8SelectColumnsOfB(B_prepared, B.nrows(), B.ncols(), cols_begin,
                            num_cols, B_prepared_selected);

  Lib::int8PrepareBias(B_prepared, A.scale(), A.zero_point(), B.scale(),
                       B.zero_point(), B.nrows(), B.ncols(), bias.data(),
                       bias_prepared);

  Lib::int8MultiplyAndAddBias(A_prepared, A.scale(), A.zero_point(),
                              B_prepared_selected, B.scale(), B.zero_point(),
                              bias_prepared, output_scale, A.nrows(), A.ncols(),
                              selected_b_layout.cols(), output);
}

TEST(IntgemmVsRuy, SelectedMultiply) {
  std::mt19937_64 gen64;
  gen64.seed(42);
  auto f = [&gen64](size_t M, size_t N, size_t P) {
    Layout a_layout(M, N, Order::RowMajor);
    Layout b_layout(N, P, Order::RowMajor);
    Layout bias_layout(1, P, Order::RowMajor);

    auto A = make_random_matrix_but_int_values(gen64, a_layout, FMIN, FMAX);
    auto B = make_random_matrix_but_int_values(gen64, b_layout, FMIN, FMAX);
    auto bias =
        make_random_matrix_but_int_values(gen64, bias_layout, FMIN, FMAX);

    std::vector<Index> cols(b_layout.cols());
    std::iota(cols.begin(), cols.end(), 0);
    std::shuffle(cols.begin(), cols.end(), gen64);
    std::uniform_int_distribution<> dist(8, cols.size());
    Index cutoff = dist(gen64);
    // The above won't do. Round to the nearest multiple of 8.
    if (cutoff % 8 != 0) {
      cutoff = static_cast<Index>(cutoff / 8) * 8;
    }

    float output_scale = 1.0f;
    Layout productLayout(M, cutoff, Order::RowMajor);
    Matrix<float> intgemmProduct(productLayout);
    MulASelectBAddBias<_Intgemm>(A, B, bias, cols.data(), cutoff,
                                 intgemmProduct.data(), output_scale);
    Matrix<float> ruyProduct(productLayout);
    MulASelectBAddBias<_Ruy>(A, B, bias, cols.data(), cutoff, ruyProduct.data(),
                             output_scale);
    float mse = MeanSquaredError(ruyProduct, intgemmProduct);
    DEBUG_PRINTABLE(ruyProduct);
    DEBUG_PRINTABLE(intgemmProduct);
    ASSERT_NEAR(mse, 0.0f, /*abs_error=*/1e-7);
  };
  run(gen64, f);
}

template <class Lib>
void MulAPreparedBQTAddBias(Matrix<float> &A,
                            Matrix<int8_t> &B_prepared_quantized_transposed,
                            Matrix<float> &bias, float *output,
                            float output_scale) {
  intgemm::AlignedVector<int8_t> mA_prepared(A.layout().num_elem()),
      mB_prepared(B_prepared_quantized_transposed.layout().num_elem());
  intgemm::AlignedVector<float> mBias_prepared(bias.layout().num_elem());

  DEBUG_PRINTABLE(A);
  DEBUG_PRINTABLE(B_prepared_quantized_transposed);
  DEBUG_PRINTABLE(bias);
  int8_t *A_prepared = mA_prepared.begin();
  int8_t *B_prepared = mB_prepared.begin();
  float *bias_prepared = mBias_prepared.begin();

  Lib::int8PrepareA(A.data(), A.scale(), A.zero_point(), A.nrows(), A.ncols(),
                    A_prepared);

  Lib::int8PrepareBFromQuantizedTransposed(
      B_prepared_quantized_transposed.data(),
      B_prepared_quantized_transposed.nrows(),
      B_prepared_quantized_transposed.ncols(), B_prepared);

  Lib::int8PrepareBias(B_prepared, A.scale(), A.zero_point(),
                       B_prepared_quantized_transposed.scale(),
                       B_prepared_quantized_transposed.zero_point(),
                       B_prepared_quantized_transposed.ncols(),
                       B_prepared_quantized_transposed.nrows(), bias.data(),
                       bias_prepared);

  Lib::int8MultiplyAndAddBias(A_prepared, A.scale(), A.zero_point(), B_prepared,

                              B_prepared_quantized_transposed.scale(),
                              B_prepared_quantized_transposed.zero_point(),

                              bias_prepared, output_scale, A.nrows(), A.ncols(),
                              B_prepared_quantized_transposed.nrows(), output);
}

TEST(IntgemmVsRuy, PrepareBFromQuantizedTransposed) {
  std::mt19937_64 gen64;
  gen64.seed(42);
  auto f = [&gen64](size_t M, size_t N, size_t P) {
    Layout a_layout(M, N, Order::RowMajor);
    Layout b_layout(N, P, Order::RowMajor);
    Layout bias_layout(1, P, Order::RowMajor);

    auto A = make_random_matrix_but_int_values(gen64, a_layout, FMIN, FMAX);
    // auto B = make_random_matrix_but_int_values(gen64, b_layout, FMIN,
    // FMAX);
    auto bias =
        make_random_matrix_but_int_values(gen64, bias_layout, FMIN, FMAX);

    auto B_quantized_transposed =
        make_random_matrix<int8_t>(gen64, b_layout.transpose(), 0, 127);

    float output_scale = 1.0f;
    Layout productLayout(M, P, Order::RowMajor);
    Matrix<float> intgemmProduct(productLayout);
    MulAPreparedBQTAddBias<_Intgemm>(A, B_quantized_transposed, bias,
                                     intgemmProduct.data(), output_scale);
    Matrix<float> ruyProduct(productLayout);
    MulAPreparedBQTAddBias<_Ruy>(A, B_quantized_transposed, bias,
                                 ruyProduct.data(), output_scale);
    float mse = MeanSquaredError(ruyProduct, intgemmProduct);
    DEBUG_PRINTABLE(ruyProduct);
    DEBUG_PRINTABLE(intgemmProduct);
    ASSERT_NEAR(mse, 0.0f, /*abs_error=*/1e-7);
  };
  run(gen64, f);
}

} // namespace

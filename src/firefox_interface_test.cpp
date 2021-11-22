#include "firefox_interface.h"
#include "matrix.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

namespace {

using namespace pg;

const float MSE_TOLERANCE = 1e-7;

// The following mechanism is to have templating. For purposes of keeping the
// functions in something global, include trickery is used. Following this, we
// place these into different namespaces so these can co-exist, not violating
// ODR.

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

#define DEBUG_PRINTABLE(x)                                                     \
  do {                                                                         \
    if (std::getenv("ARM_PLAYGROUND_DEBUG")) {                                 \
      std::cout << #x << ":\n" << x << std::endl;                              \
    }                                                                          \
  } while (0)

void run(std::mt19937_64 &gen64,
         std::function<void(size_t, size_t, size_t)> f) {
  constexpr size_t DIM_MAX = 128;
  constexpr size_t DIM_MIN = 64;
  constexpr size_t MC_RUNS = 100;
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

    // Often in debugging it's convenient to inspect what is happening with a
    // smaller matrix. The follwoing is the smallest we can get to work.
    // 1x16 A, 16x8 B and 1x8 Bias.
    // This works only with INTGEMM_CPUID=SSSE3.
    if (std::getenv("ARM_PLAYGROUND_SMALL")) {
      M = 1, N = 16, P = 8;
    }

    f(M, N, P);
  }
}


// Repeats path for lib with matrices A, B and bias. Final result goes into
// output, applied with an optional scale.
template <class Lib>
void MultiplyABAddBias(Matrix<float> &A, Matrix<float> &B, Matrix<float> &bias,
                       float *output, float output_scale) {
  Matrix<int8_t> mA_prepared(A.layout()), mB_prepared(B.layout().transpose());
  Matrix<float> mBias_prepared(bias.layout());

  int8_t *A_prepared = mA_prepared.begin();
  int8_t *B_prepared = mB_prepared.begin();
  float *bias_prepared = mBias_prepared.begin();

  // Offline, at weights construction.
  Lib::int8PrepareB(B.data(), B.scale(), B.zero_point(), B.nrows(), B.ncols(),
                    B_prepared);

  Lib::int8PrepareBias(B_prepared, A.scale(), A.zero_point(), B.scale(),
                       B.zero_point(), B.nrows(), B.ncols(), bias.data(),
                       bias_prepared);

  // The following happens online, on arrival of input, activations and imminent
  // multiply.
  Lib::int8PrepareA(A.data(), A.scale(), A.zero_point(), A.nrows(), A.ncols(),
                    A_prepared);

  Lib::int8MultiplyAndAddBias(A_prepared, A.scale(), A.zero_point(), B_prepared,
                              B.scale(), B.zero_point(), bias_prepared,
                              output_scale, A.nrows(), A.ncols(), B.ncols(),
                              output);
}

TEST(IntgemmVsRuy, NaiveMultiply) {
  std::mt19937_64 gen64;
  gen64.seed(42);
  auto f = [&gen64](size_t M, size_t N, size_t P) {
    auto [A, B, bias] = generateInput(gen64, M, N, P);

    float output_scale = 1.0f;
    DEBUG_PRINTABLE(A);
    DEBUG_PRINTABLE(B);
    DEBUG_PRINTABLE(bias);

    // Have a reference multiply ready.
    Matrix<float> refMul = ReferenceMultiply<float, float>(A, B, bias);
    DEBUG_PRINTABLE(refMul);

    Layout productLayout(M, P, Order::RowMajor);

    Matrix<float> intgemmProduct(productLayout);
    MultiplyABAddBias<_Intgemm>(A, B, bias, intgemmProduct.data(),
                                output_scale);
    float intgemm_mse = MeanSquaredError(intgemmProduct, refMul);

    DEBUG_PRINTABLE(intgemmProduct);
    ASSERT_NEAR(intgemm_mse, 0.0f, MSE_TOLERANCE);

    Matrix<float> ruyProduct(productLayout);
    MultiplyABAddBias<_Ruy>(A, B, bias, ruyProduct.data(), output_scale);

    float ruy_mse = MeanSquaredError(ruyProduct, refMul);
    DEBUG_PRINTABLE(ruyProduct);
    ASSERT_NEAR(ruy_mse, 0.0f, MSE_TOLERANCE);
  };
  run(gen64, f);
}

template <class Lib>
void MulABAddBiasWithSelect(Matrix<float> &A, Matrix<float> &B,
                            Matrix<float> &bias, Index *cols_begin,
                            Index num_cols, float *output, float output_scale) {

  // Prepare buffers to write output after prepare.
  Matrix<int8_t> mA_prepared(A.layout());

  // These are prepared once and offline.
  Matrix<int8_t> mB_prepared(B.layout().transpose());
  Matrix<float> mBias_prepared(bias.layout());

  int8_t *A_prepared = mA_prepared.begin();
  int8_t *B_prepared = mB_prepared.begin();
  float *bias_prepared = mBias_prepared.begin();

  // B is prepared offline, so is bias, at model construction.
  // Think of this as happening offline, yet together (thus emulating a single
  // function).
  Lib::int8PrepareB(B.data(), B.scale(), B.zero_point(), B.nrows(), B.ncols(),
                    B_prepared);

  Lib::int8PrepareBias(B_prepared, A.scale(), A.zero_point(), B.scale(),
                       B.zero_point(), B.nrows(), B.ncols(), bias.data(),
                       bias_prepared);

  // Things below this happen online.
  // A is activations obtained at runtime.
  Lib::int8PrepareA(A.data(), A.scale(), A.zero_point(), A.nrows(), A.ncols(),
                    A_prepared);

  // Select happens at runtime. Online.
  Matrix<int8_t> mBSelected(Layout(B.nrows(), num_cols, Order::ColMajor));

  int8_t *B_prepared_selected = mBSelected.begin();

  Lib::int8SelectColumnsOfB(B_prepared, B.nrows(), B.ncols(), cols_begin,
                            num_cols, B_prepared_selected);

  // Bias has to be selected similarly.
  auto mBias_prepared_selected =
      index_select(mBias_prepared, cols_begin, num_cols);

  float *bias_prepared_selected = mBias_prepared_selected.begin();

  Lib::int8MultiplyAndAddBias(A_prepared, A.scale(), A.zero_point(),
                              B_prepared_selected, B.scale(), B.zero_point(),
                              bias_prepared_selected, output_scale, A.nrows(),
                              A.ncols(), mBSelected.ncols(), output);
}

TEST(IntgemmVsRuy, SelectedMultiply) {
  std::mt19937_64 gen64;
  gen64.seed(42);
  auto f = [&gen64](size_t M, size_t N, size_t P) -> void {
    auto [A, B, bias] = generateInput(gen64, M, N, P);

    // Choose columns
    std::vector<Index> cols(B.ncols());
    std::iota(cols.begin(), cols.end(), 0);
    std::shuffle(cols.begin(), cols.end(), gen64);
    const int width = 8;
    std::uniform_int_distribution<> dist(width, cols.size());
    Index cutoff = dist(gen64);
    // The above won't do. Round to the nearest multiple of 8.
    if (cutoff % width != 0) {
      cutoff = static_cast<Index>(cutoff / width) * width;
    }
    std::sort(cols.begin(), cols.begin() + cutoff);

    float output_scale = 1.0f;
    Layout productLayout(M, cutoff, Order::RowMajor);

    Matrix<float> selectedB = index_select(B, cols.data(), cutoff);
    Matrix<float> selectedBias = index_select(bias, cols.data(), cutoff);
    DEBUG_PRINTABLE(A);
    DEBUG_PRINTABLE(B);
    DEBUG_PRINTABLE(selectedB);
    DEBUG_PRINTABLE(bias);
    DEBUG_PRINTABLE(selectedBias);

    // Prepare a reference multiply.
    auto refMul = ReferenceMultiply<float, float>(A, selectedB, selectedBias);
    DEBUG_PRINTABLE(refMul);

    // Test: Ruy product vs Reference
    Matrix<float> ruyProduct(productLayout);
    MulABAddBiasWithSelect<_Ruy>(A, B, bias, cols.data(), cutoff,
                                 ruyProduct.data(), output_scale);

    float ruy_mse = MeanSquaredError(ruyProduct, refMul);
    DEBUG_PRINTABLE(ruyProduct);

    ASSERT_NEAR(ruy_mse, 0.0f, MSE_TOLERANCE);

    // Test: Intgemm product vs Reference
    Matrix<float> intgemmProduct(productLayout);
    MulABAddBiasWithSelect<_Intgemm>(A, B, bias, cols.data(), cutoff,
                                     intgemmProduct.data(), output_scale);
    float intgemm_mse = MeanSquaredError(intgemmProduct, refMul);
    DEBUG_PRINTABLE(intgemmProduct);
    ASSERT_NEAR(intgemm_mse, 0.0f, MSE_TOLERANCE);
  };
  run(gen64, f);
}

template <class Lib>
void MultiplyAPreparedBQuantizedTransposedAddBias(
    Matrix<float> &A, Matrix<int8_t> &B_prepared_quantized_transposed,
    Matrix<float> &bias, float *output, float output_scale) {
  Matrix<int8_t> mA_prepared(A.layout()),
      mB_prepared(B_prepared_quantized_transposed.layout());
  Matrix<float> mBias_prepared(bias.layout());

  int8_t *A_prepared = mA_prepared.begin();
  int8_t *B_prepared = mB_prepared.begin();
  float *bias_prepared = mBias_prepared.begin();

  // Happens offline.
  Lib::int8PrepareBFromQuantizedTransposed(
      B_prepared_quantized_transposed.data(),
      B_prepared_quantized_transposed.ncols(),
      B_prepared_quantized_transposed.nrows(), B_prepared);

  Lib::int8PrepareBias(B_prepared, A.scale(), A.zero_point(),
                       B_prepared_quantized_transposed.scale(),
                       B_prepared_quantized_transposed.zero_point(),
                       B_prepared_quantized_transposed.ncols(),
                       B_prepared_quantized_transposed.nrows(), bias.data(),
                       bias_prepared);

  // Happens online.
  Lib::int8PrepareA(A.data(), A.scale(), A.zero_point(), A.nrows(), A.ncols(),
                    A_prepared);

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
    auto [A, B, bias] = generateInput(gen64, M, N, P);
    auto B_quantized_transposed =
        make_random_matrix<int8_t>(gen64, B.layout().transpose(), -8, 8);

    DEBUG_PRINTABLE(A);
    DEBUG_PRINTABLE(B_quantized_transposed);
    DEBUG_PRINTABLE(bias);
    float output_scale = 1.0f;

    Layout productLayout(M, P, Order::RowMajor);

    Matrix<float> intgemmProduct(productLayout);
    MultiplyAPreparedBQuantizedTransposedAddBias<_Intgemm>(
        A, B_quantized_transposed, bias, intgemmProduct.data(), output_scale);

    Matrix<float> ruyProduct(productLayout);
    MultiplyAPreparedBQuantizedTransposedAddBias<_Ruy>(
        A, B_quantized_transposed, bias, ruyProduct.data(), output_scale);

    // Since a reference multiply is tricky here, we simply chose to go with
    // comparing intgemm and ruy.
    float mse = MeanSquaredError(ruyProduct, intgemmProduct);
    DEBUG_PRINTABLE(ruyProduct);
    DEBUG_PRINTABLE(intgemmProduct);

    ASSERT_NEAR(mse, 0.0f, MSE_TOLERANCE);
  };
  run(gen64, f);
}

} // namespace

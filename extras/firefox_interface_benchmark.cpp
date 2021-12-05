#include "matrix.h"
#include "wrapped.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

namespace {

using namespace pg;

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

#if defined(__i386__) || defined(__x86_64__)
namespaceToStructForTemplating(Intgemm);
#endif

namespaceToStructForTemplating(Ruy);

} // namespace

template <class Lib>
double MultiplyABAddBias(Matrix<float> &A, Matrix<float> &B,
                         Matrix<float> &bias, float *output,
                         float output_scale) {
  Matrix<int8_t> mA_prepared(A.layout()), mB_prepared(B.layout().transpose());
  Matrix<float> mBias_prepared(bias.layout());

  int8_t *A_prepared = mA_prepared.begin();
  int8_t *B_prepared = mB_prepared.begin();
  float *bias_prepared = mBias_prepared.begin();

  // Offline, at weights construction.
  std::cout << "Prepare B and bias, once for the entire multiply...\n";
  Lib::int8PrepareB(B.data(), B.scale(), B.zero_point(), B.nrows(), B.ncols(),
                    B_prepared);

  Lib::int8PrepareBias(B_prepared, A.scale(), A.zero_point(), B.scale(),
                       B.zero_point(), B.nrows(), B.ncols(), bias.data(),
                       bias_prepared);

  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < 10; i++) {
    // The following happens online, on arrival of input, activations and
    // imminent multiply.
    std::cout << "Iter " << i << ": Prepare A...\n";
    Lib::int8PrepareA(A.data(), A.scale(), A.zero_point(), A.nrows(), A.ncols(),
                      A_prepared);

    std::cout << "Iter " << i << ": A*B + bias...\n";
    Lib::int8MultiplyAndAddBias(A_prepared, A.scale(), A.zero_point(),
                                B_prepared, B.scale(), B.zero_point(),
                                bias_prepared, output_scale, A.nrows(),
                                A.ncols(), B.ncols(), output);
  }
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

int main(int argc, char **argv) {
  std::mt19937_64 gen64;
  gen64.seed(42);
  size_t M = 1024, N = 1024, P = 1024;
  auto [A, B, bias] = generateInput(gen64, M, N, P);
  Matrix<float> output(Layout(M, P, Order::RowMajor));

#if defined(__i386__) || defined(__x86_64__)
  auto intgemmTime =
      MultiplyABAddBias<_Intgemm>(A, B, bias, output.begin(), 1.0f);
  std::cout << "Multiply routine (intgemm) took: " << intgemmTime << " time."
            << std::endl;
#endif

  auto ruyTime = MultiplyABAddBias<_Ruy>(A, B, bias, output.begin(), 1.0f);
  std::cout << "Multiply routine (ruy)took: " << ruyTime << " time."
            << std::endl;

  return 0;
}

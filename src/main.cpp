#include "firefox_interface.h"
#include "matrix.h"
#include <iostream>
#include <random>

using namespace pg;

#ifdef __i386__
#define Path Intgemm
#else
#define Path Ruy
#endif

void MultiplyABAddBias(Matrix<float> &A, Matrix<float> &B, Matrix<float> &bias,
                       float *output, float output_scale) {
  Matrix<int8_t> mA_prepared(A.layout()), mB_prepared(B.layout().transpose());
  Matrix<float> mBias_prepared(bias.layout());

  int8_t *A_prepared = mA_prepared.begin();
  int8_t *B_prepared = mB_prepared.begin();
  float *bias_prepared = mBias_prepared.begin();

  // Offline, at weights construction.
  Path::int8PrepareB(B.data(), B.scale(), B.zero_point(), B.nrows(), B.ncols(),
                    B_prepared);

  Path::int8PrepareBias(B_prepared, A.scale(), A.zero_point(), B.scale(),
                       B.zero_point(), B.nrows(), B.ncols(), bias.data(),
                       bias_prepared);

  // The following happens online, on arrival of input, activations and imminent
  // multiply.
  Path::int8PrepareA(A.data(), A.scale(), A.zero_point(), A.nrows(), A.ncols(),
                    A_prepared);

  Path::int8MultiplyAndAddBias(A_prepared, A.scale(), A.zero_point(), B_prepared,
                              B.scale(), B.zero_point(), bias_prepared,
                              output_scale, A.nrows(), A.ncols(), B.ncols(),
                              output);
}

int main(int argc, char **argv) {
    std::mt19937_64 gen64;
    gen64.seed(42);
    size_t M = 64, N = 128, P = 64;
    auto [A, B, bias] = generateInput(gen64, M, N, P);
    Matrix<float> output(Layout(M, P, Order::RowMajor));
    MultiplyABAddBias(A, B, bias, output.begin(), 1.0f);

    std::cout << "A:\n" << A << std::endl;
    std::cout << "B:\n" << B << std::endl;
    std::cout << "bias:\n" << bias << std::endl;

    std::cout << "output:\n" << output << std::endl;

    Matrix<float> reference = ReferenceMultiply<float, float>(A, B, bias);
    std::cout << "reference:\n" << reference << std::endl;

    return 0;
}

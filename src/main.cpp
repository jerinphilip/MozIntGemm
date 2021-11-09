#include "firefox_interface.h"
#include "matrix.h"
#include <iostream>
#include <random>

int main(int argc, char **argv) {
  // Need 8 bit ints, that's all.
  std::mt19937_64 gen64;
  gen64.seed(42);

  constexpr size_t DIM_MAX = 32;
  constexpr size_t DIM_MIN = 16;
  constexpr size_t MC_RUNS = 10000;
  for (size_t i = 0; i < MC_RUNS; i++) {
    std::uniform_int_distribution<> distribution(DIM_MIN, DIM_MAX);

    size_t M, N, P; // A = M x N matrix, B = N x P

    M = distribution(gen64);
    N = distribution(gen64);
    P = distribution(gen64);

    // Do some stuff to get stuff rounded to multiples of 8
    const size_t _WIDTH = 32;
    M = ((M / _WIDTH) + 1) * _WIDTH;
    N = ((N / _WIDTH) + 1) * _WIDTH;
    P = ((P / _WIDTH) + 1) * _WIDTH;
    // M = 1, N = 16, P = 8;
    // M = 32, N = 32, P = 32;

    using pg::Matrix;

    std::cout << M << " " << N << " " << P << "\n\n";

    Matrix<float> B(N, P), A(M, N);

    A.fill(gen64);
    B.fill(gen64, -8, 8);

    Matrix<float> biasForRuy(1, P), mBias_prepared(1, P);
    biasForRuy.fill(gen64);
    mBias_prepared.fill(biasForRuy);

    std::cout << "A:\n" << A << "\n";
    std::cout << "B:\n" << B << "\n";

    std::cout << "bias:\n" << biasForRuy << "\n";

    Matrix<float> ruyProduct(M, P), intgemmProduct(M, P);

    Matrix<int8_t> mA_prepared(M, N), mB_prepared(N, P);

    int8_t *A_prepared = mA_prepared.data();
    int8_t *B_prepared = mB_prepared.data();
    float *bias_prepared = mBias_prepared.data();

    float *output;
    output = intgemmProduct.data();

    float scale = 1.0f, zero_point = 0.0f, output_scale = 1.0f;

    pg::Intgemm::int8PrepareA(A.data(), scale, zero_point,
                              /*rows_A=*/M, /*width=*/N, A_prepared);

    pg::Intgemm::int8PrepareB(B.data(), scale, zero_point,
                              /*width=*/N, /*cols_B=*/P, B_prepared);

    pg::Intgemm::int8PrepareBias(
        B_prepared, scale, zero_point, scale, zero_point,
        /*width=*/N,
        /*cols_B=*/P, biasForRuy.data(), bias_prepared);

    pg::Intgemm::int8MultiplyAndAddBias(
        A_prepared, scale, zero_point, B_prepared, scale, zero_point,
        bias_prepared, output_scale, M, N, P, output);

    std::cout << "Intgemm A*B : \n" << intgemmProduct << "\n";

    Matrix<int8_t> BForRuy(N, P), AForRuy(M, N);
    AForRuy.fill(A);
    BForRuy.fill(B); // Make a copy so the preparation does not change opaque
                     // representations.
    output = ruyProduct.data();
    pg::Ruy::int8MultiplyAndAddBias(
        AForRuy.data(), scale, zero_point, BForRuy.data(), scale, zero_point,
        biasForRuy.data(), output_scale, M, N, P, output);

    std::cout << "Ruy A*B : \n" << ruyProduct << "\n";

    float mse = pg::MeanSquaredError(ruyProduct, intgemmProduct);
    std::cout << "Mean-Squared-Error(ruyProduct, intgemmProduct) = " << mse
              << "\n";
    float EPS = 1e-7;
    if (mse >= EPS) {
      std::cerr << "Found large MSE..." << std::endl;
      std::abort();
    }
  }
  return 0;
}

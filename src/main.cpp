#include "firefox_interface.h"
#include <iostream>
#include <memory>
#include <random>
#include <vector>

template <class T> class Matrix {
public:
  enum class Order { RowMajor, ColumnMajor };

  Matrix(size_t nrows, size_t ncols)
      : nrows_(nrows), ncols_(ncols), matrix_(nrows * ncols) {}

  T *data() { return matrix_.data(); }
  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; }

  void fill(std::mt19937_64 &gen64) {
    constexpr T _INT8_MAX = 127;
    constexpr T _INT8_MIN = -127;
    std::uniform_int_distribution<> int8_distribution(_INT8_MIN, _INT8_MAX);
    for (size_t i = 0; i < nrows_; i++) {
      for (size_t j = 0; j < ncols_; j++) {
        matrix_[i * ncols_ + j] = int8_distribution(gen64);
      }
    }
  }

  friend std::ostream &operator<<(std::ostream &out, const Matrix &matrix) {
    for (size_t i = 0; i < matrix.nrows_; i++) {
      for (size_t j = 0; j < matrix.ncols_; j++) {
        if (j != 0) {
          out << " ";
        }
        out << (int)(matrix.matrix_[i * matrix.ncols_ + j]);
      }
      out << "\n";
    }
    return out;
  }

private:
  const size_t nrows_;
  const size_t ncols_;
  std::vector<T> matrix_;
};

int main(int argc, char **argv) {
  // Need 8 bit ints, that's all.
  std::mt19937_64 gen64;
  gen64.seed(42);

  constexpr size_t DIM_MAX = 4;
  constexpr size_t DIM_MIN = 2;
  std::uniform_int_distribution<> distribution(DIM_MIN, DIM_MAX);

  size_t M, N, P; // A = M x N matrix, B = N x P

  M = distribution(gen64);
  N = distribution(gen64);
  P = distribution(gen64);

  std::cout << M << " " << N << " " << P << "\n";

  Matrix<int8_t> A(M, N), B(N, P);

  A.fill(gen64);
  B.fill(gen64);

  std::cout << "A:\n" << A;
  std::cout << "B:\n" << B;

  Matrix<float> C(M, P);

  const int8_t *A_prepared = A.data();
  const int8_t *B_prepared = B.data();
  const float *bias_prepared = nullptr;
  float *output = C.data();
  pg::Ruy::int8MultiplyAndAddBias(
      A_prepared, /*scale=*/1.0, /*zero_point=*/0.0f, B_prepared, /*scale=*/1.0,
      /*zero_point=*/0.0f, bias_prepared, /*scale_output=*/1.0f, M, N, P,
      output);

  std::cout << "C = A*B : \n" << C;

  return 0;
}

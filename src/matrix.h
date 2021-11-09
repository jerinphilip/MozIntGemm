#pragma once
#include "3rd-party/intgemm/intgemm/aligned.h"
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

namespace pg {

template <class T> class Matrix {
public:
  enum class Order { RowMajor, ColumnMajor };

  Matrix(size_t nrows, size_t ncols)
      : nrows_(nrows), ncols_(ncols), matrix_(nrows * ncols) {}

  T *data() { return matrix_.begin(); }
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

  const T &at(size_t i, size_t j) const { return matrix_[i * ncols_ + j]; }

private:
  const size_t nrows_;
  const size_t ncols_;
  intgemm::AlignedVector<T> matrix_;
};

template <class T>
float MeanSquaredError(const Matrix<T> &a, const Matrix<T> &b) {
  assert(a.nrows() == b.nrows() && a.ncols() == b.ncols());
  float mse = 0.0f;
  for (size_t i = 0; i < a.nrows(); i++) {
    for (size_t j = 0; j < a.ncols(); j++) {
      float diff = a.at(i, j) - b.at(i, j);
      mse += diff * diff;
    }
  }
  return mse;
}

} // namespace pg

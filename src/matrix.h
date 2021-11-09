#pragma once
#include "3rd-party/intgemm/intgemm/aligned.h"
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

namespace pg {

enum class Order { RowMajor, ColMajor };

template <class T> class Matrix {
public:
  Matrix(size_t nrows, size_t ncols, Order order = Order::RowMajor)
      : nrows_(nrows), ncols_(ncols), matrix_(nrows * ncols), order_(order) {}

  T *data() { return matrix_.begin(); }
  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; }

  void fill(std::mt19937_64 &gen64) {
    constexpr T _INT8_MAX = 127;
    constexpr T _INT8_MIN = -127;
    std::uniform_int_distribution<> int8_distribution(_INT8_MIN, _INT8_MAX);
    for (size_t i = 0; i < nrows_; i++) {
      for (size_t j = 0; j < ncols_; j++) {
        matrix_[address(i, j)] = int8_distribution(gen64);
      }
    }
  }

  friend std::ostream &operator<<(std::ostream &out, const Matrix &matrix) {
    for (size_t i = 0; i < matrix.nrows_; i++) {
      for (size_t j = 0; j < matrix.ncols_; j++) {
        if (j != 0) {
          out << " ";
        }
        out << (int)(matrix.matrix_[matrix.address(i, j)]);
      }
      out << "\n";
    }
    return out;
  }

  template <class OT> void fill(const Matrix<OT> &other) {
    assert(nrows() == other.nrows() and ncols() == other.ncols());
    for (size_t i = 0; i < nrows_; i++) {
      for (size_t j = 0; j < ncols_; j++) {
        matrix_[address(i, j)] = static_cast<T>(other.at(i, j));
      }
    }
  }

  const T &at(size_t i, size_t j) const { return matrix_[address(i, j)]; }

private:
  inline size_t address(size_t i, size_t j) const {
    if (order_ == Order::RowMajor) {
      return i * ncols_ + j;
    } else {
      return j * nrows_ + i;
    }
  }
  Order order_;
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

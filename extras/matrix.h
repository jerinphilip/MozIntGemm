#pragma once
#include "3rd-party/intgemm/intgemm/aligned.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

namespace pg {

using Index = std::uint32_t;

enum class Order { RowMajor, ColMajor };

// Layout allows the matrix interface below to have a layout agnostic accessor.
// It also allows to operate transposing a layout and such inorder to work out
// experimentation, tests and that sort. Consider this class as metadata for the
// Matrix.
class Layout {
public:
  Layout(size_t rows, size_t cols, Order order)
      : rows_(rows), cols_(cols), order_(order) {}

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  Order order() const { return order_; }
  inline size_t num_elem() const { return rows_ * cols_; }
  inline size_t position(size_t i, size_t j) const {
    return (order_ == Order::RowMajor) ? i * cols_ + j : j * rows_ + i;
  }

  Layout transpose() const { return Layout(cols_, rows_, order_); }

private:
  size_t rows_;
  size_t cols_;
  Order order_;
};

namespace utils {

// Utility to print matrix stored using layout. Prints for display and human
// readability. Hides elements in large matrices using ellipses, in a numpy like
// display format.
template <class Scalar>
void printMatrix(std::ostream &out, const Scalar *data, const Layout &layout);

// Convenience function to print intgemm vectors, when there is no matrix layout
// information.
template <class T>
std::ostream &operator<<(std::ostream &out, const intgemm::AlignedVector<T> &v);

} // namespace utils

// Mozilla interface uses a pointer based C-API. Intgemm is slightly better, but
// still keeps higher-level constructs minimum. For easy wiring, we hold our own
// rich-annotated matrix class from which pointers are passed for ruy/intgemm
// operations.
//
// What are properties of a matrix: something like scale to be used in
// quantization computed based on the data within the matrix, also zero point
// are installed as methods onto this class. Allows easy transposes as well.
//
// Functions here are not written with performance in mind (these are handled by
// pointer based operations in ruy/intgemm).
template <class Scalar> class Matrix {
public:
  using dtype = Scalar;
  using iterator = Scalar *;
  using const_iterator = const Scalar *;
  Matrix(const Layout &layout) : layout_(layout), matrix_(layout.num_elem()) {}
  Matrix(const Layout &layout, Scalar *data) : Matrix(layout) {
    std::memcpy(begin(), data, sizeof(Scalar) * layout.num_elem());
  }

  Matrix<Scalar> transpose();

  const Layout &layout() const { return layout_; }
  size_t nrows() const { return layout_.rows(); }
  size_t ncols() const { return layout_.cols(); }

  Scalar *data() { return matrix_.begin(); }

  iterator begin() { return data(); }
  iterator end() { return begin() + layout_.num_elem(); }

  const_iterator cbegin() const { return matrix_.begin(); }
  const_iterator cend() const { return matrix_.begin() + layout_.num_elem(); }

  friend std::ostream &operator<<(std::ostream &out, Matrix &matrix) {
    utils::printMatrix(out, matrix.data(), matrix.layout_);
    return out;
  }

  Scalar &at(size_t i, size_t j) { return matrix_[layout_.position(i, j)]; }

  const Scalar &at(size_t i, size_t j) const {
    return matrix_[layout_.position(i, j)];
  }

  float zero_point() const { return 0.0f; };

  float scale() const;

private:
  const Layout layout_;
  intgemm::AlignedVector<Scalar> matrix_;
};

// Empty template which aborts. Each type has to be specialized (int8 / float /
// int32). Sorry...
template <class Scalar>
inline Matrix<Scalar>
make_random_matrix(std::mt19937_64 &gen64, const Layout &layout,
                   const Scalar minVal, const Scalar maxVal);

// Specializations for T below, puts random values between minVal and maxVal as
// entries.

template <>
inline Matrix<int8_t>
make_random_matrix<int8_t>(std::mt19937_64 &gen64, const Layout &layout,
                           const int8_t minVal, const int8_t maxVal);

template <>
inline Matrix<int32_t>
make_random_matrix<int32_t>(std::mt19937_64 &gen64, const Layout &layout,
                            const int32_t minVal, const int32_t maxVal);

template <>
Matrix<float> inline make_random_matrix<float>(std::mt19937_64 &gen64,
                                               const Layout &layout,
                                               const float minVal,
                                               const float maxVal);

// Sometimes it's convenient to ignore floating point errors and have exact
// computational capabilities. With this function we store values that'll fit in
// the relevant part as complete integers so floating point errors don't happen.
inline Matrix<float> make_random_matrix_but_int_values(std::mt19937_64 &gen64,
                                                       const Layout &layout,
                                                       const int8_t minVal,
                                                       const int8_t maxVal);

// For testing. Intgemm's mse appears to be using 0.1f threshold in
// multiplications of matrices containing real-values in [-1.0, 1.0], for a
// guideline/reference point.
template <class Scalar>
inline float MeanSquaredError(const Matrix<Scalar> &a, const Matrix<Scalar> &b);

// MSE will do, this was installed just to check if any off-by-one sortof error
// was happening.
template <class Scalar>
inline float MaxAbsDifference(const Matrix<Scalar> &a, const Matrix<Scalar> &b);

// Index select manual, similar to torch.index_select.
template <class Scalar>
inline Matrix<Scalar> index_select(const Matrix<Scalar> &input, Index *cols,
                                   Index num_cols);

template <class Scalar, class AccumScalar>
inline Matrix<AccumScalar> ReferenceMultiply(const Matrix<Scalar> &A,
                                             const Matrix<Scalar> &B,
                                             const Matrix<Scalar> &bias);

// Generates random [A, B, bias] for computing A * B + bias in testing and
// benchmarking.
inline std::tuple<Matrix<float>, Matrix<float>, Matrix<float>>
generateInput(std::mt19937_64 &gen64, size_t M, size_t N, size_t P);

// Creates an intgeral input matrix given ordering, this is used in argmaxin
// ruy's capabilities as a function of data-access-order. Part of a failed
// experiment.
inline std::tuple<Matrix<int8_t>, Matrix<int8_t>, Matrix<int8_t>>
generateIntegralInput(std::mt19937_64 &gen64, size_t M, size_t N, size_t P,
                      const std::tuple<Order, Order, Order> &ordering);

} // namespace pg

#define PG_MATRIX_IMPL_
#include "matrix-impl.cpp"


#ifndef PG_MATRIX_IMPL_
#error "This file is not intended to be included directly!"
#endif

namespace pg {
namespace utils {

template <class Scalar>
void printMatrix(std::ostream &out, const Scalar *data, const Layout &layout) {
  const size_t truncate = 4;
  bool rowEllipses = true;
  for (size_t i = 0; i < layout.rows(); i++) {
    if (i <= truncate || layout.rows() - i <= truncate) {
      bool colEllipses = true;
      for (size_t j = 0; j < layout.cols(); j++) {
        if (j <= truncate || layout.cols() - j <= truncate) {
          if (j != 0) {
            out << " ";
          }
          out << std::fixed << std::showpoint << std::setprecision(4)
              << (double)data[layout.position(i, j)];
        } else {
          if (colEllipses)
            out << " ... ";
          colEllipses = false;
        }
      }
      out << "\n";
    } else {
      if (rowEllipses) {
        for (size_t i = 0; i < 2 * truncate + 1; i++) {
          out << "... ";
        }
        out << "\n";
      }
      rowEllipses = false;
    }
  }
}

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

} // namespace utils

template <class Scalar> Matrix<Scalar> Matrix<Scalar>::transpose() {
  Layout transposed_layout(ncols(), nrows(), layout().order());
  Matrix<Scalar> transposed(transposed_layout);
  for (size_t i = 0; i < nrows(); i++) {
    for (size_t j = 0; j < ncols(); j++) {
      transposed.at(j, i) = at(i, j);
    }
  }
  return transposed;
}

template <class Scalar> float Matrix<Scalar>::scale() const {
  // return 1.0f;
  // ^ The above is easy when setting int8_t fittable values for tests.
  Scalar maxAbsolute = 0.0;
  for (auto p = cbegin(); p != cend(); ++p) {
    maxAbsolute = std::max<Scalar>(maxAbsolute, std::abs(*p));
  }

  return 127.0f / static_cast<float>(maxAbsolute);
};

inline std::tuple<Matrix<int8_t>, Matrix<int8_t>, Matrix<int8_t>>
generateIntegralInput(std::mt19937_64 &gen64, size_t M, size_t N, size_t P,
                      const std::tuple<Order, Order, Order> &ordering) {
  auto [a_order, b_order, c_order] = ordering;
  Layout a_layout(M, N, a_order);
  Layout b_layout(N, P, b_order);
  Layout bias_layout(1, P, b_order);

  auto A = make_random_matrix<int8_t>(gen64, a_layout, -127, 127);
  auto B = make_random_matrix<int8_t>(gen64, b_layout, -127, 127);
  auto bias = make_random_matrix<int8_t>(gen64, bias_layout, -127, 127);
  return std::make_tuple(std::move(A), std::move(B), std::move(bias));
}

inline std::tuple<Matrix<float>, Matrix<float>, Matrix<float>>
generateInput(std::mt19937_64 &gen64, size_t M, size_t N, size_t P) {
  Layout a_layout(M, N, Order::RowMajor);
  Layout b_layout(N, P, Order::RowMajor);
  Layout bias_layout(1, P, Order::RowMajor);

  // The following values work for everything including SSSE3.
  // Unfortunately, to control errors, we need [-1.0f, 1.0f]. Leaving the below
  // block commented here for future multiply inspections on tiny matrices if
  // necessary).

  // auto A = make_random_matrix_but_int_values(gen64, a_layout, 0, 127);
  // auto B = make_random_matrix_but_int_values(gen64, b_layout, -8, 8);
  // auto bias = make_random_matrix_but_int_values(gen64, bias_layout, 0, 127);

  auto A = make_random_matrix<float>(gen64, a_layout, -1.0f, 1.0f);
  auto B = make_random_matrix<float>(gen64, b_layout, -1.0f, 1.0f);
  auto bias = make_random_matrix<float>(gen64, bias_layout, -1.0f, 1.0f);
  return std::make_tuple(std::move(A), std::move(B), std::move(bias));
}

template <class Scalar, class AccumScalar>
inline Matrix<AccumScalar> ReferenceMultiply(const Matrix<Scalar> &A,
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

template <class Scalar>
inline Matrix<Scalar> index_select(const Matrix<Scalar> &input, Index *cols,
                                   Index num_cols) {
  Layout layout(input.layout().rows(), num_cols, input.layout().order());
  Matrix<Scalar> selected(layout);
  for (size_t i = 0; i < input.layout().rows(); i++) {
    for (Index j = 0; j < num_cols; j++) {
      selected.at(i, j) = input.at(i, cols[j]);
    }
  }
  return selected;
}

template <class Scalar>
inline float MeanSquaredError(const Matrix<Scalar> &a,
                              const Matrix<Scalar> &b) {
  assert(a.layout().rows() == b.layout().rows() &&
         a.layout().cols() == b.layout().cols());
  float mse = 0.0f;
  for (size_t i = 0; i < a.layout().rows(); i++) {
    for (size_t j = 0; j < a.layout().cols(); j++) {
      float diff = (a.at(i, j) - b.at(i, j));
      mse += diff * diff;
    }
  }
  return std::sqrt(mse) / static_cast<float>(a.layout().num_elem());
}

template <class Scalar>
inline float MaxAbsDifference(const Matrix<Scalar> &a,
                              const Matrix<Scalar> &b) {
  assert(a.layout().rows() == b.layout().rows() &&
         a.layout().cols() == b.layout().cols());
  float maxAbsDelta = 0.0f;
  for (size_t i = 0; i < a.layout().rows(); i++) {
    for (size_t j = 0; j < a.layout().cols(); j++) {
      maxAbsDelta = std::max<float>(maxAbsDelta, (a.at(i, j) - b.at(i, j)));
    }
  }
  return maxAbsDelta;
}

template <class Scalar>
inline Matrix<Scalar>
make_random_matrix(std::mt19937_64 &gen64, const Layout &layout,
                   const Scalar minVal, const Scalar maxVal) {
  std::cerr << "Not implemented. Specialize for a type" << std::endl;
  std::abort();
}

template <>
inline Matrix<int8_t>
make_random_matrix<int8_t>(std::mt19937_64 &gen64, const Layout &layout,
                           const int8_t minVal, const int8_t maxVal) {
  Matrix<int8_t> matrix(layout);
  std::uniform_int_distribution<> int8_distribution(minVal, maxVal);
  std::generate(matrix.begin(), matrix.end(), [&gen64, &int8_distribution]() {
    return int8_distribution(gen64);
  });
  return matrix;
}

template <>
inline Matrix<int32_t>
make_random_matrix<int32_t>(std::mt19937_64 &gen64, const Layout &layout,
                            const int32_t minVal, const int32_t maxVal) {
  Matrix<int32_t> matrix(layout);
  std::uniform_int_distribution<> distribution(minVal, maxVal);
  std::generate(matrix.begin(), matrix.end(),
                [&gen64, &distribution]() { return distribution(gen64); });
  return matrix;
}

template <>
Matrix<float> inline make_random_matrix<float>(std::mt19937_64 &gen64,
                                               const Layout &layout,
                                               const float minVal,
                                               const float maxVal) {
  std::uniform_real_distribution<> real_distribution(minVal, maxVal);
  Matrix<float> matrix(layout);
  std::generate(matrix.begin(), matrix.end(), [&gen64, &real_distribution]() {
    return real_distribution(gen64);
  });
  return matrix;
}

inline Matrix<float> make_random_matrix_but_int_values(std::mt19937_64 &gen64,
                                                       const Layout &layout,
                                                       const int8_t minVal,
                                                       const int8_t maxVal) {
  std::uniform_int_distribution<> int8_distribution(minVal, maxVal);
  Matrix<float> matrix(layout);
  std::generate(matrix.begin(), matrix.end(), [&gen64, &int8_distribution]() {
    return int8_distribution(gen64);
  });
  return matrix;
}

} // namespace pg

#include "ruy/ruy.h"
#include <cassert>
#include <vector>

#ifndef PRINT_MATRIX_DEBUG
#define PRINT_MATRIX_DEBUG(d, rows, cols, order)                               \
  do {                                                                         \
  } while (0)
#endif

namespace detail {
void quantize(const float *input, float scale, float zero_point, Index rows,
              Index width, int8_t *output) {
  // Dumb quantize we will improve this eventually.
  const Index size = rows * width;
  for (size_t i = 0; i < size; i++) {
    float value = roundf(scale * input[i]);
    // int8 can't store larger than 127.0f.
    value = std::max(-127.0f, value);
    value = std::min(127.0f, value);
    output[i] = static_cast<int8_t>(value);
  };
}

template <class Scalar>
void transpose(const Scalar *input, Index rows, Index cols, Scalar *output) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      output[j * rows + i] = input[i * cols + j];
    }
  }
}

} // namespace detail

void int8PrepareB(const float *input_B, float scale, float zero_point,
                  Index width, Index cols_B, int8_t *output) {
  // Client matrix is expected to be row-major. We are allowed to change
  // internal representation starting here. Column major is preferable for B
  // when A*B (dot product of A row with B column). Ideally this function is
  // called once, offline.
  PRINT_MATRIX_DEBUG(input_B, width, cols_B, Order::RowMajor);
  std::vector<int8_t> B_quantized(width * cols_B);
  detail::quantize(input_B, scale, zero_point, width, cols_B,
                   B_quantized.data());
  PRINT_MATRIX_DEBUG(B_quantized.data(), width, cols_B, Order::RowMajor);

  // This is a lazy transpose to get overall test correct.
  // TODO(jerinphilip): Fix with optimized fixed-size transpose reuse.
  // Look for: Permutation instructions.
  detail::transpose(B_quantized.data(), width, cols_B, output);
}

void int8PrepareBFromTransposed(const float *input_B_transposed, float scale,
                                float zero_point, Index width, Index cols_B,
                                int8_t *output) {
  detail::quantize(input_B_transposed, scale, zero_point, width, cols_B,
                   output);
}

void int8PrepareBFromQuantizedTransposed(const int8_t *input_B_quant_transposed,
                                         Index width, Index cols_B,
                                         int8_t *output) {
  // Isn't this a no-op, or more specifically a copy.
  std::memcpy(output, input_B_quant_transposed,
              /*count=*/sizeof(int8_t) * (width * cols_B));
}

void int8PrepareA(const float *input_A, float scale, float zero_point,
                  Index rows_A, Index width, int8_t *output) {
  detail::quantize(input_A, scale, zero_point, rows_A, width, output);
}

void int8PrepareBias(const int8_t *input_B_prepared, float scale_A,
                     float zero_point_A, float scale_B, float zero_point_B,
                     Index width, Index cols_B, const float *input_bias,
                     float *output) {
  // Copy bias as is. Ruy supports int8_t*int8_t -> int32_t, so we don't need to
  // do any trickery with bias to add/substract offset.
  std::memcpy(output, input_bias, /*count=*/sizeof(float) * (1 * cols_B));
}

void int8MultiplyAndAddBias(const int8_t *input_A_prepared, float scale_A,
                            float zero_point_A, const int8_t *input_B_prepared,
                            float scale_B, float zero_point_B,
                            const float *input_bias_prepared,
                            float scale_output, Index rows_A, Index width,
                            Index cols_B, float *output) {

  // It is expected that somehow we have managed to call all prepare by the time
  // we are here, with inputs (prepared) in int8_t. All that's left to do is use
  // ruy for multiply and then start with the reverse ops to get to fp32.

  // Use ruy to multiply.
  // The following is adapted from
  // https://github.com/google/ruy/blob/878283640de7946a43053e8ebf4f15114fbc9156/example/example.cc#L129-L152

  ruy::Context context;
  ruy::Matrix<std::int8_t> lhs;
  ruy::MakeSimpleLayout(rows_A, width, ruy::Order::kRowMajor,
                        lhs.mutable_layout());
  lhs.set_data(input_A_prepared);

  PRINT_MATRIX_DEBUG(input_A_prepared, rows_A, width, Order::RowMajor);

  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(width, cols_B, ruy::Order::kColMajor,
                        rhs.mutable_layout());
  rhs.set_data(input_B_prepared);

  PRINT_MATRIX_DEBUG(input_B_prepared, width, cols_B, Order::ColMajor);

  ruy::Matrix<std::int32_t> dst;
  ruy::MakeSimpleLayout(rows_A, cols_B, ruy::Order::kRowMajor,
                        dst.mutable_layout());

  std::vector<std::int32_t> dst_data(rows_A * cols_B);
  std::int32_t *dest_ptr = dst_data.data();

  dst.set_data(dest_ptr);

  // When Dst is int32, mul_params is unused.
  ruy::MulParams<std::int32_t, std::int32_t> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);

  // Unquantizes, then adds bias in a single statement on the output.
  float unquant_multiplier = 1.0f * scale_output / (scale_A * scale_B);
  for (size_t i = 0; i < rows_A; i++) {
    for (size_t j = 0; j < cols_B; j++) {
      Index idx = i * cols_B + j;
      output[idx] = (static_cast<float>(dest_ptr[idx]) * unquant_multiplier) +
                    input_bias_prepared[j];
    }
  }
}

void int8SelectColumnsOfB(const int8_t *input_B_prepared, Index width,
                          Index cols_B, const Index *cols, const Index num_cols,
                          int8_t *output) {
  // B_prepared is expected to be col-major, for our implementation via ruy. If
  // col-major we can memcpy the respective column entries as they're
  // sequential. There are width=rows entries.
  for (Index c = 0; c < num_cols; ++c) {
    std::memcpy(&(output[c * width]), &(input_B_prepared[cols[c] * width]),
                width);
  }
}

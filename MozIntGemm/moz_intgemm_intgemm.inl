
void int8PrepareA(const float *input_A, float scale, float zero_point,
                  Index rows_A, Index width, int8_t *output) {
  intgemm::Int8Shift::PrepareA(input_A, output, scale, /*Quant Mult*/
                               rows_A, width);
}

void int8PrepareB(const float *input_B, float scale, float zero_point,
                  Index width, Index cols_B, int8_t *output) {
  intgemm::Int8::PrepareB(input_B, output, scale, /*Quant Mult*/
                          width, cols_B);
}

void int8PrepareBFromTransposed(const float *input_B_transposed, float scale,
                                float zero_point, Index width, Index cols_B,
                                int8_t *output) {
  intgemm::Int8::PrepareBTransposed(input_B_transposed, output, scale, width,
                                    cols_B);
}

void int8PrepareBFromQuantizedTransposed(const int8_t *input_B_quant_transposed,
                                         Index width, Index cols_B,
                                         int8_t *output) {
  intgemm::Int8::PrepareBQuantizedTransposed(input_B_quant_transposed, output,
                                             width, cols_B);
}

void int8PrepareBias(const int8_t *input_B_prepared, float scale_A,
                     float zero_point_A, float scale_B, float zero_point_B,
                     Index width, Index cols_B, const float *input_bias,
                     float *output) {
  float unquant_factor =
      (-1) * ((127.0f / scale_A) * (127.0f / scale_B)) / (127.0f);
  intgemm::Int8Shift::PrepareBias(
      input_B_prepared, width, cols_B,
      intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_factor,
                                                       input_bias, output));
}

void int8MultiplyAndAddBias(const int8_t *input_A_prepared, float scale_A,
                            float zero_point_A, const int8_t *input_B_prepared,
                            float scale_B, float zero_point_B,
                            const float *input_bias_prepared,
                            float unquant_multiplier, Index rows_A, Index width,
                            Index cols_B, float *output) {
  float unquant_factor = unquant_multiplier / (scale_A * scale_B);
  intgemm::Int8Shift::Multiply(
      input_A_prepared, input_B_prepared, rows_A, width, cols_B,
      intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
          unquant_factor, input_bias_prepared, output));
}

void int8SelectColumnsOfB(const int8_t *input_B_prepared, Index width,
                          Index cols_B, const Index *cols, const Index num_cols,
                          int8_t *output) {
  intgemm::Int8::SelectColumnsB(input_B_prepared, output, width, cols,
                                cols + num_cols);
}

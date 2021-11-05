#include "3rd_party/wasm_intgemm/wasm_intgemm_interface.h"
#include "ruy/ruy.h"

void int8PrepareB(const float *input_B, float scale, float zero_point,
                  Index width, Index cols_B, int8_t *output) {}

void int8PrepareBFromTransposed(const float *input_B_transposed, float scale,
                                float zero_point, Index width, Index cols_B,
                                int8_t *output) {}

void int8PrepareBFromQuantizedTransposed(const int8_t *input_B_quant_transposed,
                                         Index width, Index cols_B,
                                         int8_t *output) {}

void int8PrepareA(const float *input_A, float scale, float zero_point,
                  Index rows_A, Index width, int8_t *output) {}

void int8PrepareBias(const int8_t *input_B_prepared, float scale,
                     float zero_point, Index width, Index cols_B,
                     const float *input_bias, float *output) {}

void int8MultiplyAndAddBias(const int8_t *input_A_prepared, float scale_A,
                            float zero_point_A, const int8_t *input_B_prepared,
                            float scale_B, float zero_point_B,
                            const float *input_bias_prepared, Index rows_A,
                            Index width, Index cols_B, float *output) {}

void int8SelectColumnsOfB(const int8_t *input_B_prepared, Index width,
                          Index cols_B, const Index *cols, const Index num_cols,
                          int8_t *output) {}

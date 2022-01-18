namespace detail {

template <class T> class AlignedVector {
public:
  AlignedVector(size_t num_elem)
      : size_(num_elem),
        storage_(reinterpret_cast<T *>(
            ruy::detail::SystemAlignedAlloc(sizeof(T) * num_elem))) {}

  T *begin() { return storage_; }
  T *data() { return storage_; }
  size_t size() const { return size_; }
  size_t memSize() const { return sizeof(T) * size_; }

  // Forbid copy
  AlignedVector(const AlignedVector &) = delete;
  AlignedVector &operator=(const AlignedVector &) = delete;

  ~AlignedVector() {
    ruy::detail::SystemAlignedFree(reinterpret_cast<void *>(storage_));
  }

private:
  T *storage_;
  size_t size_;
};

// TODO: Workout similar to ruy. enum value is causing type/value complaints.
// Intentions are to get the functions to exist together without flouting
// One-Definition-Rule (ODR).
struct kStandardCpp {};
struct kNeon {};

#if RUY_PLATFORM_NEON
using kHighestPath = kNeon;
#else
using kHighestPath = kStandardCpp;
#endif

template <class Path> struct Preprocess {
  static void quantize(const float *input, float scale, float zero_point,
                       Index rows, Index width, int8_t *output) {
    const Index size = rows * width;
    for (size_t i = 0; i < size; i++) {
      // Round to nearest after multiplying with scale.
      float value = roundf(scale * input[i]);

      // Since float can store bigger values, we threshold anything that's gone
      // higher and can't fit in int8.
      value = std::max<float>(-127.0f, value);
      value = std::min<float>(127.0f, value);

      // Finally a static cast.
      output[i] = static_cast<int8_t>(value);
    };
  }

  template <class Scalar>
  static void transpose(const Scalar *input, Index rows, Index cols,
                        Scalar *output) {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        output[j * rows + i] = input[i * cols + j];
      }
    }
  }
  static void unquantizeAddBias(const int32_t *input,
                                const float *input_bias_prepared,
                                float unquant_multiplier, Index rows_A,
                                Index cols_B, float *output) {
    for (size_t i = 0; i < rows_A; i++) {
      for (size_t j = 0; j < cols_B; j++) {
        Index idx = i * cols_B + j;
        output[idx] =
            (input[idx] * unquant_multiplier) + input_bias_prepared[j];
      }
    }
  }
};

#if RUY_PLATFORM_NEON
template <> struct Preprocess<kNeon> {
  static void quantize(const float *input, float scale, float zero_point,
                       Index rows, Index width, int8_t *output) {
    const float32x4_t *Input = reinterpret_cast<const float32x4_t *>(input);
    const float32x4_t *InputEnd =
        reinterpret_cast<const float32x4_t *>(input + rows * width);

    int8x8_t *Output = reinterpret_cast<int8x8_t *>(output);
    while (Input != InputEnd) {
      // Vector multiply by scalar
      // float32x4_t vmulq_n_f32(float32x4_t a, float32_t b);
      // VMUL.F32 q0,q0,d0[0]
      float32x4_t scaledFloat_lo = vmulq_n_f32(*Input++, scale);

      // Convert from float
      // int32x4_t  vcvtnq_s32_f32(float32x4_t a);
      // VCVT.S32.F32 q0, q0
      int32x4_t scaledInt_lo = vcvtnq_s32_f32(scaledFloat_lo);

      // Vector saturating narrow integer
      // int16x4_t  vqmovn_s32(int32x4_t a);   // VQMOVN.S32 d0,q0
      int16x4_t s16x4_lo = vqmovn_s32(scaledInt_lo);

      // Vector multiply by scalar
      // float32x4_t vmulq_n_f32(float32x4_t a, float32_t b);
      // VMUL.F32 q0,q0,d0[0]
      float32x4_t scaledFloat_hi = vmulq_n_f32(*Input++, scale);

      // Convert from float
      // int32x4_t  vcvtnq_s32_f32(float32x4_t a);
      // VCVT.S32.F32 q0, q0
      int32x4_t scaledInt_hi = vcvtnq_s32_f32(scaledFloat_hi);

      // Vector saturating narrow integer
      // int16x4_t  vqmovn_s32(int32x4_t a);
      // VQMOVN.S32 d0,q0
      int16x4_t s16x4_hi = vqmovn_s32(scaledInt_hi);

      // Combine two ints.
      // int16x8_t   vcombine_s16(int16x4_t low, int16x4_t high);
      int16x8_t s16x8 = vcombine_s16(s16x4_lo, s16x4_hi);

      // Vector saturating narrow integer
      int8x8_t s8x8 = vqmovn_s16(s16x8);

      *Output = s8x8;
      ++Output;
    };
  }

  template <class Scalar>
  static void transpose(const Scalar *input, Index rows, Index cols,
                        Scalar *output) {
    // This is a template with an abort. The specialized implementation is done
    // below for int8_t.
    std::abort();
  }

  // Specialization for int8_t
  static void transpose(const int8_t *input, Index rows, Index cols,
                        int8_t *output) {
    constexpr size_t tile_size = 16;
    // TODO(jerin): Enable
    // assert(rows % tile_size == 0 && cols & tile_size == 0);
    for (size_t i = 0; i < rows; i += tile_size) {
      for (size_t j = 0; j < cols; j += tile_size) {
        _transpose_16x16(input, i, j, rows, cols, output);
      }
    }
  }

  static void _transpose_16x16(const int8_t *src, Index i, Index j, Index rows,
                               Index cols, int8_t *dst) {
    // Implemented following the algorithm described in
    // https://stackoverflow.com/a/29587984/4565794
    //
    // permute n 32-bit rows
    // permute n 64-bit rows
    // ...
    // permute n simd_width/2-bit rows

    // clang-format off
    
    // Permute 8 8-bit rows.
    // Load int8x16x2 from memory into SIMD registers, transpose as 2x2 matrices.
    Index rowBegin = i*cols + j;

    int8x16x2_t r0 = vtrnq_s8(vld1q_s8(&src[ 0*cols + rowBegin]), vld1q_s8(&src[ 1*cols + rowBegin]));
    int8x16x2_t r1 = vtrnq_s8(vld1q_s8(&src[ 2*cols + rowBegin]), vld1q_s8(&src[ 3*cols + rowBegin]));
    int8x16x2_t r2 = vtrnq_s8(vld1q_s8(&src[ 4*cols + rowBegin]), vld1q_s8(&src[ 5*cols + rowBegin]));
    int8x16x2_t r3 = vtrnq_s8(vld1q_s8(&src[ 6*cols + rowBegin]), vld1q_s8(&src[ 7*cols + rowBegin]));
    int8x16x2_t r4 = vtrnq_s8(vld1q_s8(&src[ 8*cols + rowBegin]), vld1q_s8(&src[ 9*cols + rowBegin]));
    int8x16x2_t r5 = vtrnq_s8(vld1q_s8(&src[10*cols + rowBegin]), vld1q_s8(&src[11*cols + rowBegin]));
    int8x16x2_t r6 = vtrnq_s8(vld1q_s8(&src[12*cols + rowBegin]), vld1q_s8(&src[13*cols + rowBegin]));
    int8x16x2_t r7 = vtrnq_s8(vld1q_s8(&src[14*cols + rowBegin]), vld1q_s8(&src[15*cols + rowBegin]));


    // Permute 8 16-bit rows.
    // Next step is to treat the entries as int16x8x2 (via cast) and do
    // transpose for int16, which will now leave intra-2 pairs intact while
    // transposing inter 2-pairs into the right places.
    int16x8x2_t t0 = vtrnq_s16(vreinterpretq_s16_s8(r0.val[0]), vreinterpretq_s16_s8(r1.val[0]));
    int16x8x2_t t1 = vtrnq_s16(vreinterpretq_s16_s8(r2.val[0]), vreinterpretq_s16_s8(r3.val[0]));
    int16x8x2_t t2 = vtrnq_s16(vreinterpretq_s16_s8(r4.val[0]), vreinterpretq_s16_s8(r5.val[0]));
    int16x8x2_t t3 = vtrnq_s16(vreinterpretq_s16_s8(r6.val[0]), vreinterpretq_s16_s8(r7.val[0]));
    int16x8x2_t t4 = vtrnq_s16(vreinterpretq_s16_s8(r0.val[1]), vreinterpretq_s16_s8(r1.val[1]));
    int16x8x2_t t5 = vtrnq_s16(vreinterpretq_s16_s8(r2.val[1]), vreinterpretq_s16_s8(r3.val[1]));
    int16x8x2_t t6 = vtrnq_s16(vreinterpretq_s16_s8(r4.val[1]), vreinterpretq_s16_s8(r5.val[1]));
    int16x8x2_t t7 = vtrnq_s16(vreinterpretq_s16_s8(r6.val[1]), vreinterpretq_s16_s8(r7.val[1]));

    // Permute 8 32-bit rows.
    int32x4x2_t x0 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[0]), vreinterpretq_s32_s16(t1.val[0]));
    int32x4x2_t x1 = vtrnq_s32(vreinterpretq_s32_s16(t4.val[0]), vreinterpretq_s32_s16(t5.val[0]));
    int32x4x2_t x2 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[1]), vreinterpretq_s32_s16(t1.val[1]));
    int32x4x2_t x3 = vtrnq_s32(vreinterpretq_s32_s16(t4.val[1]), vreinterpretq_s32_s16(t5.val[1]));

    int32x4x2_t x4 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[0]), vreinterpretq_s32_s16(t3.val[0]));
    int32x4x2_t x5 = vtrnq_s32(vreinterpretq_s32_s16(t6.val[0]), vreinterpretq_s32_s16(t7.val[0]));
    int32x4x2_t x6 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[1]), vreinterpretq_s32_s16(t3.val[1]));
    int32x4x2_t x7 = vtrnq_s32(vreinterpretq_s32_s16(t6.val[1]), vreinterpretq_s32_s16(t7.val[1]));

    // There is no permute 8 64-bit rows available. 
    // Instead we follow extracting low and high and placing them into the right places.
    Index tgtRowBegin = j*rows + i;
    vst1q_s8(&dst[ 0*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x0.val[0]), vget_low_s32(x4.val[0])))); 
    vst1q_s8(&dst[ 1*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x1.val[0]), vget_low_s32(x5.val[0]))));
    vst1q_s8(&dst[ 2*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x2.val[0]), vget_low_s32(x6.val[0]))));
    vst1q_s8(&dst[ 3*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x3.val[0]), vget_low_s32(x7.val[0]))));
    vst1q_s8(&dst[ 4*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x0.val[1]), vget_low_s32(x4.val[1]))));
    vst1q_s8(&dst[ 5*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x1.val[1]), vget_low_s32(x5.val[1]))));
    vst1q_s8(&dst[ 6*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x2.val[1]), vget_low_s32(x6.val[1]))));
    vst1q_s8(&dst[ 7*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(x3.val[1]), vget_low_s32(x7.val[1]))));

    vst1q_s8(&dst[ 8*rows + tgtRowBegin], vreinterpretq_s8_s32 (vcombine_s32(vget_high_s32(x0.val[0]), vget_high_s32(x4.val[0]))));
    vst1q_s8(&dst[ 9*rows + tgtRowBegin], vreinterpretq_s8_s32 (vcombine_s32(vget_high_s32(x1.val[0]), vget_high_s32(x5.val[0]))));
    vst1q_s8(&dst[10*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x2.val[0]), vget_high_s32(x6.val[0]))));
    vst1q_s8(&dst[11*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x3.val[0]), vget_high_s32(x7.val[0]))));
    vst1q_s8(&dst[12*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x0.val[1]), vget_high_s32(x4.val[1]))));
    vst1q_s8(&dst[13*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x1.val[1]), vget_high_s32(x5.val[1]))));
    vst1q_s8(&dst[14*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x2.val[1]), vget_high_s32(x6.val[1]))));
    vst1q_s8(&dst[15*rows + tgtRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x3.val[1]), vget_high_s32(x7.val[1]))));

    // clang-format on
  }

  static void unquantizeAddBias(const int32_t *input,
                                const float *input_bias_prepared,
                                float unquant_multiplier, Index rows_A,
                                Index cols_B, float *output) {
    // Set all registers in lane from same scalar value.
    float32x4_t multiplier = vdupq_n_f32(unquant_multiplier);
    const int32x4_t *Input = reinterpret_cast<const int32x4_t *>(input);
    const int32x4_t *InputEnd =
        reinterpret_cast<const int32x4_t *>(input + rows_A * cols_B);
    float32x4_t *Output = reinterpret_cast<float32x4_t *>(output);

    while (Input != InputEnd) {
      // Bias cycles every column for addition.
      const float32x4_t *Bias =
          reinterpret_cast<const float32x4_t *>(input_bias_prepared);

      // InputEnd needs to be determined to end the while loop below.
      const int32x4_t *RowEnd = reinterpret_cast<const int32x4_t *>(
          reinterpret_cast<const int32_t *>(Input) + cols_B);

      while (Input != RowEnd) {
        // Operation happening for 4-elements together:
        // output = [int32_t]input * [float]quant_mult + [float]bias;
        float32x4_t floatInput = vcvtq_f32_s32(*Input++);
        float32x4_t unquantized = vmulq_f32(floatInput, multiplier);
        *Output++ = vaddq_f32(unquantized, *Bias++);
      }
    }
  }
};
#endif
} // namespace detail

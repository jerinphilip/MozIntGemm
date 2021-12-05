/** A fallback (non-optimized) implementation of "wasm_gemm_interface.h"
 * interface for integer matrix multiplication for wasm target.
 *
 * This implementation is built and exported from the main module and can serve
 * as a polyfill (fallback) for browsers that don't support an optimized
 * implementation of "wasm_gemm_interface.h".
 */

#include "3rd-party/intgemm/intgemm/intgemm.h"
#include "moz_intgemm.h"
#include <iostream>

#include "moz_intgemm_intgemm.inl"

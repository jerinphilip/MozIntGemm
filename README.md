# MozIntGemm


![armv8](https://github.com/jerinphilip/MozIntGemm/actions/workflows/arm-compiles.yml/badge.svg)
![x86-tests](https://github.com/jerinphilip/MozIntGemm/actions/workflows/tests.yml/badge.svg)

Formerly arm-playground.

## Premise

Neural models involve a lot of matrix multiplies. Speeding up the matrix
multiplication provides us with fast neural network inference on the client
machine.

<img width="300" alt="The pile gets soaked with data and starts to get mushy over time, so it's technically recurrent." src="https://imgs.xkcd.com/comics/machine_learning.png">

The matrix-multiplications as compiled by emscripten from C++ sources into
WebAssembly target turned out to be too slow.

From [Bugzilla/1746631#c1](https://bugzilla.mozilla.org/show_bug.cgi?id=1746631#c1):

> * Wasm Gemm : 95 wps 
> * Wormhole : 390 wps (+310% to Wasm Gemm)
> * Native Firefox gemm
>   * SSSE3 : 490 wps (+25% to Wormhole, +415% to Wasm Gemm)
>   * AVX2 : 560 wps (+43% to Wormhole, +489% to Wasm Gemm)

While the original support was intended for x86-64 based architectures through
[kpu/intgemm](https://github.com/kpu/intgemm), and the above-mentioned
interface is written closely bound to
[kpu/intgemm](https://github.com/kpu/intgemm), an immediate demand surfaced for
support for ARM.

A target-agnostic matrix multiply interface suitable for WASM was checked in by
collaborators at Mozilla for purposes of speeding up translations in
https://github.com/browsermt/marian-dev/pull/49. This repository takes the
interface provided above and attempts to provide an implementation for the
interface using [google/ruy](https://github.com/google/ruy/) in the short term.

The library target which uses either ruy or intgemm depending on the platform
is made available as `moz_intgemm`. Most of the sources are in
[MozIntGemm](./MozIntGemm) directory.

## Testing and Benchmarking

The interface is available for testing closer to the real use case [only on the
WASM
platform](https://github.com/browsermt/marian-dev/blob/08b1544636fe13eaf1fbacb17c6fb050abfb8d42/src/tensors/cpu/integer_common.h#L8).
This creates difficulty in doing end-to-end checks closer to the original task.

Fortunately, to test correctness, all we have to do is assert that a certain
sequence of operations used through
[kpu/intgemm](https://github.com/kpu/intgemm)
implementation gives the same output when used through
[google/ruy](https://github.com/google/ruy). A translation workload for marian
is a composition of the functions specified in the Mozilla interface.

Testing for correctness happens on the x86 backend, that both ruy and intgemm
support.  When intgemm is turned off on ARM (where it lacks support) and ruy
allowed to take over, we thus obtain an ARM backend for bergamot-translator's
matrix multiplies, w.r.t integration with firefox and fast local translation
extension.

There is also a slow reference implementation vs fast SIMD implementation
using NEON intrinsics which covers source-code added here as preprocessing
functions (quantize, unquantize, transpose).


## LICENSE

The additions here are MIT Licensed. Respective licenses provided by original
source in submodules inside [3rd-party/](./3rd-party) apply.


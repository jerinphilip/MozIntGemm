# arm-playground

![armv8](https://github.com/jerinphilip/arm-playground/actions/workflows/arm-compiles.yml/badge.svg)
![x86-tests](https://github.com/jerinphilip/arm-playground/actions/workflows/tests.yml/badge.svg)


## Objective

A target-agnostic matrix multiply interface suitable for WASM was checked in by
collaborators at Mozilla for purposes of speeding up translations in
https://github.com/browsermt/marian-dev/pull/49.

While the original support was intended for x86-64 based architectures through
[kpu/intgemm](https://github.com/kpu/intgemm), and the above mentioned
interface written closely bound to
[kpu/intgemm](https://github.com/kpu/intgemm), a demand surfaced for support for ARM.

This repository takes the interface provided by Mozilla and attempts to provide an
implementation for the interface using
[google/ruy](https://github.com/google/ruy/) in the short-term.

The library target which uses either ruy or intgemm depending on platform is
available as `moz_intgemm`. Most of the sources are in
[MozIntGemm](./MozIntGemm) directory.

## Testing and Benchmarking

A translation-workload for marian is a composition of the functions specified
in the Mozilla interface. To check correctness, all we have to do is a certain
sequence of operations used through
[kpu/intgemm](https://github.com/kpu/intgemm) implementation gives the same
output when used through [google/ruy](https://github.com/google/ruy).

Testing for correctness happens on the x86 backend, which ruy and intgemm
support.  When intgemm is turned off on ARM (where it lacks support) and ruy
allowed to take over, we get an ARM backend for bergamot-translator's matrix multiplies.

### Status

| Function                            | Correctness | Optimized for performance(?) |
| ----------------------------------- | ------------| --------------------------   |
| int8PrepareB                        | Yes         | Yes                          |
| int8PrepareBFromTransposed          | Yes         | Yes                          |
| int8PrepareBFromQuantizedTransposed | Yes         | Yes                          |
| int8PrepareA                        | Yes         | Yes                          |
| int8PrepareBias                     | Yes         | Yes                          |
| int8SelectColumsOfB                 | Yes         | Yes                          |
| int8MultiplyAndAddBias              | Yes         | Yes                          |


## relatable xkcd comics 

<img width="600" alt="How to mess with people who’ve learned to expect rounding errors in floating-point math." src="https://imgs.xkcd.com/comics/e_to_the_pi_minus_pi.png">

<img width="300" alt="The pile gets soaked with data and starts to get mushy over time, so it's technically recurrent." src="https://imgs.xkcd.com/comics/machine_learning.png">


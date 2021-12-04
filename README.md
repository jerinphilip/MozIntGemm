# arm-playground

![armv8](https://github.com/jerinphilip/arm-playground/actions/workflows/arm-compiles.yml/badge.svg)
![x86-tests](https://github.com/jerinphilip/arm-playground/actions/workflows/tests.yml/badge.svg)


Need to support a backend for arm for bergamot-translator. Easy way is to
integrate arm support into [intgemm](https://github.com/kpu/intgemm), copying
over stuff from [ruy](https://github.com/google/ruy). This repository is meant
to help with the task... or something. 

Objective: ASSERT(Multiply-By-Ruy = Multiply-By-Intgemm), on x86. These will work as tests here.

* [src/firefox\_interface.inl](src/firefox_interface.inl): Interface dictated by firefox for compatibility with something inside firefox.
* [src/firefox\_interface.h](src/firefox_interface.h): Duplicates interface specified by firefox to be implemented by ruy and intgemm both.
* [src/impl\_ruy-export.cpp](src/impl_ruy-export.cpp): Implementation of the above interface in ruy.
* [src/impl\_intgemm.cpp](src/impl_intgemm.cpp): Implementation of the above interface in intgemm. This is adapted from [wasm\_fallback\_interface.cpp](https://github.com/browsermt/marian-dev/blob/master/src/tensors/cpu/wasm_intgemm_fallback.cpp) provided by Abhishek Aggarwal.

Once things work on x86, we'll simply turn off intgemm which doesn't compile on
ARM and the remaining ruy which is agnostic to x86 or ARM will work and provide
an implementation of the firefox dictated API.

## relatable xkcd comics 

<img width="300" alt="The pile gets soaked with data and starts to get mushy over time, so it's technically recurrent." src="https://imgs.xkcd.com/comics/machine_learning.png">

<img width="300" alt="How to mess with people who’ve learned to expect rounding errors in floating-point math." src="https://imgs.xkcd.com/comics/e_to_the_pi_minus_pi.png">

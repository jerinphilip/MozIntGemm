# arm-playground

Need to support a backend for arm for bergamot-translator. Easy way is to integrate arm support into [intgemm](https://github.com/kpu/intgemm), copying over stuff from [ruy](https://github.com/google/ruy). This repository is meant to help with the task... or something. 

Objective: ASSERT(Multiply-By-Ruy = Multiply-By-Intgemm), on x86. These will work as tests here.

* [src/firefox\_interface.inl](src/firefox_interface.inl): Interface dictated by firefox for compatibility with something inside firefox.
* [src/firefox\_interface.h](src/firefox_interface.h): Duplicates interface specified by firefox to be implemented by ruy and intgemm both.
* [src/impl\_ruy.cpp](src/impl_ruy.cpp): Implementation of the above interface in ruy.
* [src/impl\_ruy.cpp](src/impl_ruy.cpp): Implementation of the above interface in ruy.

![image](https://user-images.githubusercontent.com/727292/139909229-7648899c-1d97-4fc7-9def-a310aa815da9.png)

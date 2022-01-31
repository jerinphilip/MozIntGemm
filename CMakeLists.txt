cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(arm-playground CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adapted from https://stackoverflow.com/a/14934542/4565794
# Not probably best, but okay for time being. It's ruy that imposes these constraints.
# https://github.com/google/ruy/blob/8c3fd3f266b4a22d542d4aa41329b5018d6b87e1/ruy/platform.h#L100-L127
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Insufficient gcc version")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "1920")
    message(FATAL_ERROR "Insufficient msvc version")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "CLANG")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.0")
    message(FATAL_ERROR "Insufficient clang version")
  endif()
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

option(COMPILE_BENCHMARKS "Compile benchmarks." OFF)
option(COMPILE_TESTS "Compile tests." OFF)


message(${CMAKE_SYSTEM_PROCESSOR})
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  set(BUILD_ARCH_ARM ON)
else()
  set(BUILD_ARCH_ARM OFF)
endif()

add_subdirectory(3rd-party)

if(BUILD_ARCH_ARM)
    set(EXT_LIBS cpuinfo ruy)
else() 
    set(EXT_LIBS cpuinfo ruy intgemm)
endif()

if(COMPILE_TESTS)
  enable_testing()
endif(COMPILE_TESTS)

add_subdirectory(MozIntGemm)

if(COMPILE_TESTS)
    add_subdirectory(extras)
endif(COMPILE_TESTS)

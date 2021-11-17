cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(arm-playground CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(${CMAKE_ANDROID_ARCH_ABI} MATCHES "arm")
  set(BUILD_ARCH_ARM ON)
else()
  set(BUILD_ARCH_ARM FALSE)
endif()

add_subdirectory(3rd-party)

if(BUILD_ARCH_ARM)
    set(COMPILE_TESTS OFF)
    set(EXT_LIBS cpuinfo ruy)
else() 
    set(EXT_LIBS cpuinfo ruy intgemm)
endif()

if(COMPILE_TESTS)
  enable_testing()
endif(COMPILE_TESTS)

add_subdirectory(src)

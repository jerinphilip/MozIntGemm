cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(arm-playground CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")


message(${CMAKE_SYSTEM_PROCESSOR})
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  set(BUILD_ARCH_ARM ON)
else()
  set(BUILD_ARCH_ARM OFF)
endif()

add_subdirectory(3rd-party)

if(BUILD_ARCH_ARM)
    set(COMPILE_TESTS OFF CACHE BOOL "No tests for ARM, oops" FORCE)
    set(EXT_LIBS cpuinfo ruy)
else() 
    set(EXT_LIBS cpuinfo ruy intgemm)
endif()

if(COMPILE_TESTS)
  enable_testing()
endif(COMPILE_TESTS)

add_subdirectory(src)

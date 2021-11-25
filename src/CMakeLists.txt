
if(BUILD_ARCH_ARM)
    set(SOURCES impl_ruy.cpp)
else(BUILD_ARCH_ARM)
    set(SOURCES impl_ruy.cpp impl_intgemm.cpp)
endif(BUILD_ARCH_ARM)

add_library(bridge STATIC ${SOURCES})
target_link_libraries(bridge ${EXT_LIBS})
target_include_directories(bridge PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(main main.cpp)
target_link_libraries(main bridge)

if(COMPILE_TESTS)
  add_executable(firefox_interface_test firefox_interface_test.cpp)
  target_link_libraries(firefox_interface_test gtest_main gtest bridge)
  include(GoogleTest)
  gtest_discover_tests(firefox_interface_test)
endif(COMPILE_TESTS)

add_executable(benchmark_empirical_multiply benchmark_empirical_multiply.cpp)
target_link_libraries(benchmark_empirical_multiply bridge)



add_library(bridge STATIC impl_ruy.cpp impl_intgemm.cpp)
target_link_libraries(bridge ruy cpuinfo intgemm)
target_include_directories(bridge PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(main main.cpp)
target_link_libraries(main bridge)

if(COMPILE_TESTS)
  enable_testing()
  add_executable(firefox_interface_test firefox_interface_test.cpp)
  target_link_libraries(firefox_interface_test gtest_main bridge)
  # gtest_discover_tests(firefox_interface_test)
endif(COMPILE_TESTS)

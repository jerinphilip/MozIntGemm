add_library(bridge STATIC wrapped.cpp)
target_link_libraries(bridge ${EXT_LIBS})
target_include_directories(bridge PUBLIC ${CMAKE_SOURCE_DIR})

if(COMPILE_TESTS)
  if(BUILD_ARCH_ARM)
    add_executable(detail_test detail_test.cpp)
    target_link_libraries(detail_test gtest_main gtest bridge)
  else(BUILD_ARCH_ARM)
    add_executable(firefox_interface_test firefox_interface_test.cpp)
    target_link_libraries(firefox_interface_test gtest_main gtest bridge)
  endif(BUILD_ARCH_ARM)

  include(GoogleTest)

  if(BUILD_ARCH_ARM)
    gtest_discover_tests(detail_test)
  else(BUILD_ARCH_ARM)
    gtest_discover_tests(firefox_interface_test)
  endif(BUILD_ARCH_ARM)
endif(COMPILE_TESTS)

add_executable(firefox_interface_benchmark firefox_interface_benchmark.cpp)
target_link_libraries(firefox_interface_benchmark bridge)


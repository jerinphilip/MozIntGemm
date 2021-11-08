
add_library(bridge STATIC impl_ruy.cpp impl_intgemm.cpp)
target_link_libraries(bridge ruy cpuinfo intgemm)
target_include_directories(bridge PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(main main.cpp)
target_link_libraries(main bridge)

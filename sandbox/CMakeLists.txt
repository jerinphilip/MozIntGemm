
add_library(bridge STATIC ruy_impl.cpp intgemm_impl.cpp)
target_link_libraries(bridge ruy intgemm)
target_include_directories(bridge PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(main main.cpp)
target_link_libraries(main cpuinfo bridge)

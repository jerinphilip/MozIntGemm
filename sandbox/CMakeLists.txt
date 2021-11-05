
add_executable(ruy_main ruy_main.cpp)
target_link_libraries(ruy_main ruy cpuinfo)

add_executable(main main.cpp)
target_link_libraries(main ruy cpuinfo)

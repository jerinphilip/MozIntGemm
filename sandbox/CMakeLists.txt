
add_library(bridge STATIC wasm_interface_arm.cpp wasm_intgemm_fallback.cpp)
target_link_libraries(bridge ruy intgemm)
target_include_directories(bridge PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(ruy_main ruy_main.cpp)
target_link_libraries(ruy_main cpuinfo bridge)

add_executable(main main.cpp)
target_link_libraries(main cpuinfo bridge)

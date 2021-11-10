
# Why? cpuinfo... :-|
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL " " FORCE)
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL " " FORCE)
set(CPUINFO_BUILD_PKG_CONFIG OFF CACHE BOOL " " FORCE)
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL " " FORCE)
set(CPUINFO_BUILD_TOOLS      OFF CACHE BOOL " " FORCE)


add_subdirectory(ruy/third_party/cpuinfo EXCLUDE_FROM_ALL)
add_subdirectory(ruy EXCLUDE_FROM_ALL)
add_subdirectory(ruy/third_party/googletest EXCLUDE_FROM_ALL)
add_subdirectory(intgemm EXCLUDE_FROM_ALL)

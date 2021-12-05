
if(BUILD_ARCH_ARM)
    set(SOURCES moz_intgemm_ruy.cpp)
    set(EXT_LIBS ruy)
else(BUILD_ARCH_ARM)
    set(SOURCES moz_intvemm_intgemm.cpp)
    set(EXT_LIBS intgemm)
endif(BUILD_ARCH_ARM)

add_library(moz_intgemm STATIC ${SOURCES})
target_link_libraries(moz_intgemm ${EXT_LIBS})
target_include_directories(moz_intgemm PUBLIC ${CMAKE_SOURCE_DIR})

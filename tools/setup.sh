
#!/bin/bash

NDK="/home/jerin/builds/android-ndk-r23b"
ABI="arm64-v8a"
MINSDK_VERSION=21


OTHER_ARGS=(
    -DANDROID_ARM_NEON=TRUE
)

cmake \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ABI \
    -DANDROID_NATIVE_API_LEVEL=$MINSDKVERSION \
    "${OTHER_ARGS[@]}" \
    ..

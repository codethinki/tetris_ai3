# Official dlib integration — same pattern as examples/CMakeLists.txt:
# https://github.com/davisking/dlib/blob/master/examples/CMakeLists.txt

include(FetchContent)

set(DLIB_USE_CUDA OFF CACHE BOOL "" FORCE)
set(USE_AVX2_INSTRUCTIONS ON CACHE BOOL "" FORCE)
set(USE_AVX_INSTRUCTIONS OFF CACHE BOOL "" FORCE)
set(DLIB_USE_BLAS OFF CACHE BOOL "" FORCE)
set(DLIB_USE_LAPACK oFF CACHE BOOL "" FORCE)
set(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "" FORCE)
set(DLIB_PNG_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_JPEG_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_GIF_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_WEBP_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_LINK_WITH_SQLITE3 OFF CACHE BOOL "" FORCE)
set(DLIB_USE_FFTW OFF CACHE BOOL "" FORCE)
set(DLIB_USE_MKL_FFT OFF CACHE BOOL "" FORCE)
set(DLIB_USE_FFMPEG OFF CACHE BOOL "" FORCE)
set(DLIB_USE_MKL_SEQUENTIAL OFF CACHE BOOL "" FORCE)
set(DLIB_USE_MKL_WITH_TBB OFF CACHE BOOL "" FORCE)

# dlib v20.0.1 calls enable_language(CUDA) before it sets archs; CMake 4.x requires
# CMAKE_CUDA_ARCHITECTURES to be non-empty for that probe.
if(DLIB_USE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA GPU architectures for dlib" FORCE)
endif()

FetchContent_Declare(
    dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG v20.0.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dlib)

cth_assert_target(
    dlib
    REASON "dlib target was not created — check CUDA/cuDNN configure output above"
)

if(NOT TARGET dlib::dlib)
    add_library(dlib::dlib ALIAS dlib)
endif()


cth_target_enable_build_cache(dlib OPTIONAL)

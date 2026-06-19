# Legacy FindCUDA shim for dlib on CMake 4+ (Kitware removed FindCUDA.cmake).
# Loaded by dlib's find_package(CUDA 7.5 MODULE).

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/cuda_legacy_macros.cmake)

if(CUDA_FOUND)
    if(NOT CUDA_INCLUDE_DIRS)
        find_package(CUDAToolkit REQUIRED)
        set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    endif()
    return()
endif()

find_package(CUDAToolkit REQUIRED)

set(CUDA_FOUND TRUE)
set(CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_LIBRARY_DIR}")
set(CUDA_NVCC_EXECUTABLE "${CUDAToolkit_NVCC_EXECUTABLE}")

if(NOT CUDA_NVCC_EXECUTABLE)
    find_program(
        CUDA_NVCC_EXECUTABLE
        NAMES nvcc nvcc.exe
        HINTS ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_LIBRARY_DIR}/../bin
        REQUIRED
    )
endif()

find_library(
    CUDA_CUBLAS_LIBRARIES
    NAMES cublas
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib/x64
    REQUIRED
)
find_library(
    CUDA_curand_LIBRARY
    NAMES curand
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib/x64
    REQUIRED
)
find_library(
    CUDA_cusolver_LIBRARY
    NAMES cusolver
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib/x64
    REQUIRED
)
find_library(
    CUDA_CUDART_LIBRARY
    NAMES cudart
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib/x64
    REQUIRED
)

if(NOT CMAKE_CUDA_COMPILER AND CUDA_NVCC_EXECUTABLE)
    set(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

mark_as_advanced(
    CUDA_CUBLAS_LIBRARIES
    CUDA_CUDART_LIBRARY
    CUDA_curand_LIBRARY
    CUDA_cusolver_LIBRARY
    CUDA_NVCC_EXECUTABLE
    CUDA_TOOLKIT_ROOT_DIR
)

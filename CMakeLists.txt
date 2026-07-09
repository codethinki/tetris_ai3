cmake_minimum_required(VERSION 4.0.0)
project(
  tetris_ai
  LANGUAGES CXX
  VERSION 1.0.0
)

include(cth_setup_utilities)
include(cth_target_utilities)
include(cth_install_utilities)

cth_set_compiler_specifics()
cth_set_latest_c_cpp_standard()
set(CMAKE_CXX_SCAN_FOR_MODULES FALSE)

set(CMAKE_C_COMPILER_LAUNCHER "buildcache")
set(CMAKE_CXX_COMPILER_LAUNCHER "buildcache")


# subprojects
add_subdirectory(ta3_sim)
add_subdirectory(ta3_ai)
add_subdirectory(ta3_trainer)

set(TA3_GPU_BACKEND "CUDA" CACHE STRING "GPU model-evaluator backend: CUDA or SYCL")
set_property(CACHE TA3_GPU_BACKEND PROPERTY STRINGS CUDA SYCL)

add_subdirectory(ta3_gpu)

# test
if(TA3_ENABLE_TESTS)
	message(STATUS "enabled testing")
	enable_testing()
endif()


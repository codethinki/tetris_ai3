cmake_minimum_required(VERSION 4.1.1)
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

set(TA3_GPU_BACKEND "CUDA" CACHE STRING "GPU model-evaluator backend: CUDA or SYCL")
set_property(CACHE TA3_GPU_BACKEND PROPERTY STRINGS CUDA SYCL)

# test, before the subdirectories: only directories added after enable_testing() register their tests
if(TA3_ENABLE_TESTS)
	message(STATUS "enabled testing")
	enable_testing()
endif()

# dependencies, EXCLUDE_FROM_ALL: only the cth targets ta3 links get built
set(TA3_CTH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/cth" CACHE PATH "cth sources, point it at a local clone to develop both")
# ta3 links cth, cth::os and cth::coro
set(CTH_BUILD_NET OFF)
set(CTH_BUILD_WIN OFF)
add_subdirectory("${TA3_CTH_DIR}" "${CMAKE_BINARY_DIR}/lib/cth" EXCLUDE_FROM_ALL)

# subprojects
add_subdirectory(ta3_sim)
add_subdirectory(ta3_ai)
add_subdirectory(ta3_trainer)

add_subdirectory(ta3_gpu)

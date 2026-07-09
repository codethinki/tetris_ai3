cmake_minimum_required(VERSION 4.0)
# subproject to compile with adaptive cpp
project(ta3_gpu_sycl LANGUAGES CXX VERSION 1.0.0)

include(cth_target_utilities)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES FALSE)

# name the archive ta3_gpu_sycl.lib (COFF) so the MSVC linker consumes it directly
set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".lib")

find_package(AdaptiveCpp CONFIG REQUIRED)

if(NOT ACPP_TARGETS)
    set(ACPP_TARGETS "generic")
endif()

# glob sources locally -- add/remove .cpp files and they are picked up automatically
cth_glob(TA3_GPU_SYCL_SRC "../../src/sycl" PATTERNS "*.cpp")

add_library(ta3_gpu_sycl STATIC ${TA3_GPU_SYCL_SRC})

# the sim/ai/gpu headers handed down by the superbuild (-DTA3_SIM_INCL / -DTA3_AI_INCL / -DTA3_GPU_INCL)
target_include_directories(ta3_gpu_sycl PRIVATE "${TA3_SIM_INCL}" "${TA3_AI_INCL}" "${TA3_GPU_INCL}")

# disable MSVC STL's manually-vectorized algorithms in the device TU. they compile to out-of-line
# runtime helpers (__std_mismatch_*, __std_swap_ranges_*, ...) that are host-only, so any std::equal /
# std::array ==/<=>/swap reached from a kernel leaves an unresolved extern that ptxas can't link. with
# this off the STL uses inline header loops -> device-clean. host-side SIMD is irrelevant here (this lib
# is the device codegen), and it stays self-consistent within this subproject.
target_compile_definitions(ta3_gpu_sycl PRIVATE _USE_STD_VECTOR_ALGORITHMS=0)

# ignore its prefix from here on -> Boost resolves from vcpkg
# (newest), exactly like the parent MSVC build does.
list(APPEND CMAKE_IGNORE_PREFIX_PATH "${AdaptiveCpp_ROOT}")

find_package(cth CONFIG REQUIRED)

target_link_libraries(ta3_gpu_sycl PRIVATE cth::cth)

# THE sycl compile. clean because this project's compiler is clang++ (GNU driver),
# matching AdaptiveCpp's flag style.
add_sycl_to_target(TARGET ta3_gpu_sycl SOURCES ${TA3_GPU_SYCL_SRC})

# install the archive; the adapter picks it up from the install prefix by path.
install(TARGETS ta3_gpu_sycl ARCHIVE DESTINATION lib)

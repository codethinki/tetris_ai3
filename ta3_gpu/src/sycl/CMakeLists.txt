include(ExternalProject)

# MSVC side only needs the acpp runtime import lib to link against.
find_package(AdaptiveCpp CONFIG REQUIRED)

# the bundled clang++ always lives in the acpp install tree AdaptiveCpp_ROOT points at
set(ACPP_CLANGXX "${AdaptiveCpp_ROOT}/bin/clang++.exe" CACHE FILEPATH "AdaptiveCpp bundled clang++")

set(_dev_install "${CMAKE_CURRENT_BINARY_DIR}/sycl-install")
set(_dev_lib     "${_dev_install}/lib/ta3_gpu_sycl.lib")

# adds the sycl project. The subproject configures itself from its own preset
# (adaptivecpp/CMakePresets.json), which supplies the toolchain, AdaptiveCpp/cth roots and
# ACPP_TARGETS. We only pass the superbuild-owned wiring the child can't know on its own:
# the acpp clang++, the install prefix, and the sim/ai/gpu include dirs.
ExternalProject_Add(ta3_gpu_sycl_adaptivecpp
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../toolchains/adaptivecpp"
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        --preset acpp-$<LOWER_CASE:$<CONFIG>>
        -S <SOURCE_DIR>
        -B <BINARY_DIR>
        -DCMAKE_CXX_COMPILER=${ACPP_CLANGXX}
        -DCMAKE_INSTALL_PREFIX=${_dev_install}
        -DTA3_SIM_INCL=${CMAKE_SOURCE_DIR}/ta3_sim/incl
        -DTA3_AI_INCL=${CMAKE_SOURCE_DIR}/ta3_ai/incl
        -DTA3_GPU_INCL=${CMAKE_CURRENT_SOURCE_DIR}/../../incl
    BUILD_BYPRODUCTS "${_dev_lib}"
    INSTALL_DIR "${_dev_install}"
    BUILD_ALWAYS TRUE
)

# imported handle for the acpp-built static lib
add_library(ta3_gpu_sycl_static STATIC IMPORTED GLOBAL)
set_target_properties(ta3_gpu_sycl_static PROPERTIES IMPORTED_LOCATION "${_dev_lib}")
add_dependencies(ta3_gpu_sycl_static ta3_gpu_sycl_adaptivecpp)

# the public target the parent links: the imported lib + acpp runtime + sim/ai headers. no incl dir of its
# own -- the public GPU header lives in the parent ta3_gpu target.
add_library(ta3_gpu_sycl INTERFACE)
target_link_libraries(ta3_gpu_sycl INTERFACE ta3_gpu_sycl_static AdaptiveCpp::acpp-rt ta3::sim ta3::ai)
add_dependencies(ta3_gpu_sycl ta3_gpu_sycl_adaptivecpp)

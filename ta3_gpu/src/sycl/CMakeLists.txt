include(ExternalProject)

# MSVC side only needs the acpp runtime import lib to link against.
find_package(AdaptiveCpp CONFIG REQUIRED)

# the bundled clang++ always lives in the acpp install tree AdaptiveCpp_ROOT points at
set(ACPP_CLANGXX "${AdaptiveCpp_ROOT}/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}" CACHE FILEPATH "AdaptiveCpp bundled clang++")


# acpp deploy constants
set(_acpp_script "${AdaptiveCpp_ROOT}/bin/acpp")
set(_acpp_deploy_dir "${CMAKE_CURRENT_BINARY_DIR}/acpp_deploy")
set(_acpp_deploy_components "core,cuda")

# acpp on windows python, on linux shebang
if(WIN32)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_acpp_deploy_command "${Python3_EXECUTABLE}" "${_acpp_script}" "--acpp-deploy=${_acpp_deploy_components}:${_acpp_deploy_dir}")
else()
    set(_acpp_deploy_command "${_acpp_script}" "--acpp-deploy=${_acpp_deploy_components}:${_acpp_deploy_dir}")
endif()

# deploy acpp deps at configure time
if(NOT EXISTS "${_acpp_deploy_dir}")
    message(STATUS "ta3_gpu_sycl: deploying AdaptiveCpp runtime redistributables to '${_acpp_deploy_dir}' ...")

    file(MAKE_DIRECTORY "${_acpp_deploy_dir}")

    execute_process(
        COMMAND ${_acpp_deploy_command}
        RESULT_VARIABLE _acpp_deploy_result
        OUTPUT_VARIABLE _acpp_deploy_stdout
        ERROR_VARIABLE _acpp_deploy_stderr
    )

    if(NOT _acpp_deploy_result EQUAL 0)
        # remove the partial dir so the next configure retries instead of silently skipping
        file(REMOVE_RECURSE "${_acpp_deploy_dir}")
        message(FATAL_ERROR
            "ta3_gpu_sycl: 'acpp --acpp-deploy' failed (exit ${_acpp_deploy_result}).\n"
            "stdout:\n${_acpp_deploy_stdout}\nstderr:\n${_acpp_deploy_stderr}"
        )
    endif()

    file(GLOB_RECURSE _acpp_deploy_check_files LIST_DIRECTORIES FALSE "${_acpp_deploy_dir}/*")
    if(NOT _acpp_deploy_check_files)
        # remove the empty dir so the next configure retries instead of silently skipping
        file(REMOVE_RECURSE "${_acpp_deploy_dir}")
        message(FATAL_ERROR
            "ta3_gpu_sycl: 'acpp --acpp-deploy' exited 0 but '${_acpp_deploy_dir}' is empty.\n"
            "stdout:\n${_acpp_deploy_stdout}\nstderr:\n${_acpp_deploy_stderr}"
        )
    endif()

endif()

set(_dev_install "${CMAKE_CURRENT_BINARY_DIR}/sycl-install")
set(_dev_lib     "${_dev_install}/lib/ta3_gpu_sycl.lib")

# adds the sycl project, needed bc of different compiler
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

# the public target the parent links: the imported lib + acpp runtime + sim/ai headers. no incl dir of
# its own -- the public GPU header lives in the parent ta3_gpu target.
add_library(ta3_gpu_sycl INTERFACE)
target_link_libraries(ta3_gpu_sycl INTERFACE ta3_gpu_sycl_static AdaptiveCpp::acpp-rt ta3::sim ta3::ai)
add_dependencies(ta3_gpu_sycl ta3_gpu_sycl_adaptivecpp)

# attach deploy files
cth_glob(_acpp_deploy_files "${_acpp_deploy_dir}" PATTERNS "*" PRESERVE_SUBPATHS)

# attach transitive deps acpp's deploy manifests miss
file(GLOB _acpp_extra_deps "${AdaptiveCpp_ROOT}/bin/libomp.dll" "$ENV{CUDA_PATH}/bin/cudart64_*.dll")
if(NOT _acpp_extra_deps)
    message(WARNING "ta3_gpu_sycl: found neither libomp.dll nor cudart64_*.dll -- backend plugins will fail to load (error 126) on machines without them on PATH")
endif()

cth_target_attach_copy_dependency(ta3_gpu_sycl ${_acpp_deploy_files} ${_acpp_extra_deps})

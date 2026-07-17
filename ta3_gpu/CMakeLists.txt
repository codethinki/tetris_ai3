cth_glob_cpp(TA3_GPU_INCL "incl")

add_library(ta3_gpu STATIC src/common/model_eval_host.cpp ${TA3_GPU_INCL})
add_library(ta3::gpu ALIAS ta3_gpu)

target_include_directories(ta3_gpu PUBLIC incl)

find_package(cth CONFIG REQUIRED COMPONENTS cth)
target_link_libraries(ta3_gpu PUBLIC cth::cth ta3::sim ta3::ai)

cth_target_enable_build_cache(ta3_gpu OPTIONAL)

# exactly one backend per binary, selected at link time by TA3_GPU_BACKEND (set on the root CMakeLists.txt).
if(TA3_GPU_BACKEND STREQUAL "CUDA")
	add_subdirectory(src/cuda)
	target_link_libraries(ta3_gpu PUBLIC ta3_gpu_cuda)
elseif(TA3_GPU_BACKEND STREQUAL "SYCL")
	add_subdirectory(src/sycl)
	target_link_libraries(ta3_gpu PUBLIC ta3_gpu_sycl)
else()
	message(FATAL_ERROR "TA3_GPU_BACKEND must be CUDA or SYCL (got '${TA3_GPU_BACKEND}')")
endif()

if(TA3_ENABLE_TESTS)
	add_subdirectory(tests)
endif()

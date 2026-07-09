# CUDA 13.2+ recommended
if(NOT CMAKE_CUDA_FLAGS MATCHES "allow-unsupported-compiler")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler" CACHE STRING "nvcc flags" FORCE)
endif()

enable_language(CUDA)

file(GLOB_RECURSE TA3_GPU_CUDA_CU CONFIGURE_DEPENDS "*.cu")
file(GLOB_RECURSE TA3_GPU_CUDA_CPP CONFIGURE_DEPENDS "*.cpp")

add_library(ta3_gpu_cuda STATIC ${TA3_GPU_CUDA_CU} ${TA3_GPU_CUDA_CPP})

cth_target_enable_build_cache(ta3_gpu_cuda OPTIONAL)

# no public incl of its own -- the public header lives in the parent ta3_gpu target.
target_include_directories(ta3_gpu_cuda PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../incl")

find_package(cth CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(ta3_gpu_cuda PUBLIC cth::cth ta3::sim ta3::ai CUDA::cudart)

set_target_properties(
        ta3_gpu_cuda
        PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES native
)

# the search core is constexpr and pulled into device code, so relaxed constexpr is required.
target_compile_options(ta3_gpu_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_compile_definitions(ta3_gpu_cuda PRIVATE DISABLE_CPP23)

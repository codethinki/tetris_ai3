cth_glob_cpp(TA3_GPU_TESTS_SRC "src")

add_executable(ta3_gpu_tests ${TA3_GPU_TESTS_SRC})

cth_target_enable_build_cache(ta3_gpu_tests OPTIONAL)
target_include_directories(ta3_gpu_tests PRIVATE src)

# search_move keeps its depth-3 frontier buffers (stack_vector<…, FRONTIER_CAP>) as function locals, so
# one call needs ~FRONTIER_CAP * sizeof(search_node) of stack. bump past MSVC's 1 MB default until the
# frontier moves to global scratch (design doc §5).
if(MSVC)
    target_link_options(ta3_gpu_tests PRIVATE /STACK:8388608)
endif()


find_package(GTest CONFIG REQUIRED)

# this exe is built by the normal MSVC toolchain -- ta3::gpu brings the backend-specific static lib +
# runtime + sim/ai headers.
target_link_libraries(ta3_gpu_tests PRIVATE GTest::gtest_main ta3::gpu)

include(GoogleTest)
gtest_discover_tests(ta3_gpu_tests)

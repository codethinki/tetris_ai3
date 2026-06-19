cth_glob_cpp(TA3_AI_TESTS_SRC "src")

add_executable(ta3_ai_tests ${TA3_AI_TESTS_SRC})

cth_target_enable_build_cache(ta3_ai_tests OPTIONAL)
target_include_directories(
        ta3_ai_tests
        PRIVATE src
)


find_package(GTest CONFIG REQUIRED)

# ModelV4 tests use dlib CUDA inference (same runtime requirement as ta3_ai).
target_link_libraries(ta3_ai_tests PRIVATE GTest::gtest_main ta3::ai)

include(GoogleTest)
gtest_discover_tests(ta3_ai_tests)

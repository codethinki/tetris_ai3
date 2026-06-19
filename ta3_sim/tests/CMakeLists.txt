cth_glob_cpp(TA3_SIM_TESTS_SRC "src")

add_executable(ta3_sim_tests ${TA3_SIM_TESTS_SRC})

cth_target_enable_build_cache(ta3_sim_tests OPTIONAL)
target_include_directories(
        ta3_sim_tests
        PRIVATE src
)


find_package(GTest CONFIG REQUIRED)

target_link_libraries(ta3_sim_tests PRIVATE GTest::gtest_main ta3::sim)

include(GoogleTest)
gtest_discover_tests(ta3_sim_tests)

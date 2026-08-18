cth_glob_cpp(TA3_TRAINER_TESTS_SRC "src")

add_executable(ta3_trainer_tests ${TA3_TRAINER_TESTS_SRC})

cth_target_enable_build_cache(ta3_trainer_tests OPTIONAL)
target_include_directories(
        ta3_trainer_tests
        PRIVATE src
)


find_package(GTest CONFIG REQUIRED)

target_link_libraries(
        ta3_trainer_tests
        PRIVATE
        GTest::gtest_main
        ta3::trainer
)

include(GoogleTest)
gtest_discover_tests(ta3_trainer_tests)

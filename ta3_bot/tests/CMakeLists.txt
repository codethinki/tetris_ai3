cth_glob_cpp(TA3_BOT_TESTS_SRC "src")

add_executable(ta3_bot_tests ${TA3_BOT_TESTS_SRC})

cth_target_enable_build_cache(ta3_bot_tests OPTIONAL)
target_include_directories(
        ta3_bot_tests
        PRIVATE src
)

find_package(GTest CONFIG REQUIRED)

target_link_libraries(ta3_bot_tests PRIVATE GTest::gtest_main ta3::bot)

target_compile_definitions(
        ta3_bot_tests
        PRIVATE TA3_TEST_MODEL_FILE="${PROJECT_SOURCE_DIR}/models/v11_16_32_32_score_296.champ"
)

include(GoogleTest)
gtest_discover_tests(ta3_bot_tests)

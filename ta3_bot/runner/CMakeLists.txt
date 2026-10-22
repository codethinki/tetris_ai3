cth_glob_cpp(TA3_BOT_RUNNER_SRC "src")

add_executable(ta3_bot_runner ${TA3_BOT_RUNNER_SRC})
add_executable(ta3::bot_runner ALIAS ta3_bot_runner)

cth_target_enable_build_cache(ta3_bot_runner OPTIONAL)

target_link_libraries(
        ta3_bot_runner
        PRIVATE
        ta3::bot
)

target_compile_definitions(
        ta3_bot_runner
        PRIVATE TA3_DEFAULT_MODEL_FILE="${PROJECT_SOURCE_DIR}/models/survival_improved.champ"
)

cth_target_copy_dependencies(ta3_bot_runner)

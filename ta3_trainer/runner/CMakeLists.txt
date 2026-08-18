cth_glob_cpp(TA3_TRAINER_RUNNER_SRC "src")

add_executable(ta3_trainer_runner ${TA3_TRAINER_RUNNER_SRC})
add_executable(ta3::trainer_runner ALIAS ta3_trainer_runner)

cth_target_enable_build_cache(ta3_trainer_runner OPTIONAL)

target_link_libraries(
        ta3_trainer_runner
        PRIVATE
        ta3::trainer
)

cth_target_copy_dependencies(ta3_trainer_runner)

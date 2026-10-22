cth_glob_cpp(TA3_BOT_SRC "src")
cth_glob_cpp(TA3_BOT_INCL "incl")

add_library(ta3_bot STATIC ${TA3_BOT_SRC} ${TA3_BOT_INCL})
add_library(ta3::bot ALIAS ta3_bot)

cth_target_enable_build_cache(ta3_bot OPTIONAL)


target_include_directories(
        ta3_bot
        PUBLIC incl
)


target_link_libraries(
        ta3_bot
        PUBLIC
        ta3::ai
        ta3::sim
        cth::cth
        cth::coro
        cth::net
        cth::win
)

add_subdirectory(runner)

# tests
if(TA3_ENABLE_TESTS)
    add_subdirectory(tests)
endif()

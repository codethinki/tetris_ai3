cth_glob_cpp(TA3_AI_SRC "src")
cth_glob_cpp(TA3_AI_INCL "incl")

add_library(ta3_ai STATIC ${TA3_AI_SRC} ${TA3_AI_INCL})
add_library(ta3::ai ALIAS ta3_ai)

cth_target_enable_build_cache(ta3_ai OPTIONAL)


target_include_directories(
        ta3_ai
        PUBLIC incl
)


find_package(cth CONFIG REQUIRED)

target_link_libraries(ta3_ai PUBLIC cth::cth ta3::sim dlib::dlib)

# tests
if(TA3_ENABLE_TESTS)
    add_subdirectory(tests)
endif()

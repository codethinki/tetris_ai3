cth_glob_cpp(TA3_SIM_SRC "src")
cth_glob_cpp(TA3_SIM_INCL "incl")

add_library(ta3_sim STATIC ${TA3_SIM_SRC} ${TA3_SIM_INCL})
add_library(ta3::sim ALIAS ta3_sim)

cth_target_enable_build_cache(ta3_sim OPTIONAL)


target_include_directories(
        ta3_sim
        PUBLIC incl
)


find_package(cth CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(raylib CONFIG REQUIRED)

target_link_libraries(ta3_sim PUBLIC cth::cth glm::glm raylib)

# tests
if(TA3_ENABLE_TESTS)
    add_subdirectory(tests)
endif()

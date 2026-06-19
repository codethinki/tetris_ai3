cth_glob_cpp(TA3_TRAINER_SRC "src")
cth_glob_cpp(TA3_TRAINER_INCL "incl")

add_executable(ta3_trainer ${TA3_TRAINER_SRC} ${TA3_TRAINER_INCL})
add_executable(ta3::trainer ALIAS ta3_trainer)

cth_target_enable_build_cache(ta3_trainer OPTIONAL)


target_include_directories(
        ta3_trainer
        PRIVATE incl
)


find_package(cth CONFIG REQUIRED)
find_package(pagmo CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Eigen3 CONFIG REQUIRED)

target_link_libraries(
        ta3_trainer
        PRIVATE
        ta3::ai
        Pagmo::pagmo
        Boost::serialization
        Eigen3::Eigen
)

if(MSVC)
    target_compile_options(ta3_trainer PRIVATE /openmp)
else()
    target_compile_options(ta3_trainer PRIVATE -fopenmp)
endif()

# tests (none yet — ta3::trainer is an executable)
# if(TA3_ENABLE_TESTS)
#     add_subdirectory(tests)
# endif()

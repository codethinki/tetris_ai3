cth_glob_cpp(TA3_TRAINER_SRC "src")
cth_glob_cpp(TA3_TRAINER_INCL "incl")

add_library(ta3_trainer STATIC ${TA3_TRAINER_SRC} ${TA3_TRAINER_INCL})
add_library(ta3::trainer ALIAS ta3_trainer)

cth_target_enable_build_cache(ta3_trainer OPTIONAL)


target_include_directories(
        ta3_trainer
        PUBLIC incl
)


find_package(cth CONFIG REQUIRED COMPONENTS cth coro)
find_package(pagmo CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Eigen3 CONFIG REQUIRED)
find_package(zstd CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(
        ta3_trainer
        PUBLIC
        ta3::ai
        ta3::gpu
        Pagmo::pagmo
        Boost::serialization
        cth::coro
        cth::cth
        OpenMP::OpenMP_CXX
        PRIVATE
        zstd::libzstd
)

add_subdirectory(runner)

if(TA3_ENABLE_TESTS)
    add_subdirectory(tests)
endif()

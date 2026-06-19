cmake_minimum_required(VERSION 4.0.0)
project(
  tetris_ai
  LANGUAGES CXX
  VERSION 1.0.0
)

include(cth_setup_utilities)
include(cth_target_utilities)
include(cth_install_utilities)

# dlib uses nvcc; configure it before global /std:c++latest is applied.
add_subdirectory(lib/dlib)

cth_set_compiler_specifics()
cth_set_latest_c_cpp_standard()
set(CMAKE_CXX_SCAN_FOR_MODULES FALSE)

set(CMAKE_C_COMPILER_LAUNCHER "buildcache")
set(CMAKE_CXX_COMPILER_LAUNCHER "buildcache")


# subprojects
add_subdirectory(ta3_sim)
add_subdirectory(ta3_ai)
add_subdirectory(ta3_trainer)

# test
if(TA3_ENABLE_TESTS)
	message(STATUS "enabled testing")
	enable_testing()
endif()

# formatting
cth_glob_cpp(
  FORMAT_FILES
  "ta3_ai"
  "ta3_sim"
  "ta3_trainer"
)

#cth_add_uncrustify_target(uncrustify-format OPTIONAL ${FORMAT_FILES})

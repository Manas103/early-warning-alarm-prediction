cmake_minimum_required(VERSION 3.16)
project(early_warning_edge_scorer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

include_directories(include)

add_executable(edge_scorer src/scorer_main.cpp)
add_executable(oracle_diff tests/oracle_diff_main.cpp)
add_executable(test_model tests/test_model.cpp)

if(ENABLE_ASAN_UBSAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

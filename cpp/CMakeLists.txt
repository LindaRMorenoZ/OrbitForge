cmake_minimum_required(VERSION 3.16)
project(orbitforge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(orbitforge_core src/orbitforge.cpp)
target_include_directories(orbitforge_core PUBLIC include)
target_compile_options(orbitforge_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(orbitforge_demo examples/demo.cpp)
target_link_libraries(orbitforge_demo PRIVATE orbitforge_core)

enable_testing()
add_executable(orbitforge_tests tests/test_core.cpp)
target_link_libraries(orbitforge_tests PRIVATE orbitforge_core)
add_test(NAME orbitforge_tests COMMAND orbitforge_tests)

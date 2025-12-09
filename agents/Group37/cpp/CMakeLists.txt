cmake_minimum_required(VERSION 3.10)
project(hex_mcts)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for release
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")

# Debug flags with sanitizers
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fsanitize=undefined -Wall -Wextra")

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

# If pybind11 not found via config, try downloading it
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Source files
set(SOURCES
    board.cpp
    mcts_node.cpp
    heuristics.cpp
    mcts.cpp
    transposition_table.cpp
    bindings.cpp
)

# Create Python module
pybind11_add_module(libhex_mcts ${SOURCES})

# Link libraries if needed
target_include_directories(libhex_mcts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Installation
install(TARGETS libhex_mcts LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)

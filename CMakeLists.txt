cmake_minimum_required(VERSION 3.16)
project(GradientDescent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========= pybind11 Setup =========
find_package(Python COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz
)
FetchContent_MakeAvailable(pybind11)

# ========= Source Files =========
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

# Remove main.cpp if accidentally placed in src/
list(FILTER SOURCES EXCLUDE REGEX ".*/main.cpp")

# ========= Library =========
add_library(GDLib ${SOURCES})

target_include_directories(GDLib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ========= Python Binding Module =========
pybind11_add_module(gradient_descent 
    bindings.cpp
    ${SOURCES}  # Include your source files in the binding
)

target_include_directories(gradient_descent
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against your library if needed
target_link_libraries(gradient_descent PRIVATE GDLib)

# ========= Main Executable =========
add_executable(GradientDescent main.cpp)
target_link_libraries(GradientDescent GDLib)

# ========= GoogleTest Setup =========
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

# ========= Tests =========
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS test/*.cpp)

if(TEST_SOURCES)
    add_executable(GD_Tests ${TEST_SOURCES})
    target_include_directories(GD_Tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(GD_Tests
        PRIVATE
        GTest::gtest_main
        GDLib
    )
    include(GoogleTest)
    gtest_discover_tests(GD_Tests)
else()
    message("No test sources found in 'test/' directory — skipping GD_Tests target.")
endif()

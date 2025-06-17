cmake_minimum_required(VERSION 3.16)
project(GradientDescent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========= Source Files =========
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/autodiff/main.cpp)

add_library(GDLib ${SOURCES})

target_include_directories(GDLib
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ========= Main Executable =========
add_executable(GradientDescent src/autodiff/main.cpp)
target_link_libraries(GradientDescent GDLib)

# ========= GoogleTest Setup =========
include(FetchContent)
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

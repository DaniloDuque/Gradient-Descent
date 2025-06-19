cmake_minimum_required(VERSION 3.16)
project(GradientDescent VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)

# Fetch pybind11 if not available
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz
)
FetchContent_MakeAvailable(pybind11)

# === Source files ===

# Autodiff sources (all .cpp under src/autodiff recursively)
file(GLOB_RECURSE AUTODIFF_SOURCES CONFIGURE_DEPENDS src/autodiff/*.cpp)

# Other project sources, excluding main.cpp and notebooks
file(GLOB_RECURSE OTHER_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(FILTER OTHER_SOURCES EXCLUDE REGEX ".*/main.cpp")
list(FILTER OTHER_SOURCES EXCLUDE REGEX ".*/notebooks/.*")

# Combine all library sources
set(LIB_SOURCES ${AUTODIFF_SOURCES} ${OTHER_SOURCES})

# === Library target ===
add_library(GDLib STATIC ${LIB_SOURCES})

target_include_directories(GDLib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# === Executable target ===
add_executable(GradientDescent main.cpp)

target_include_directories(GradientDescent
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(GradientDescent
    PRIVATE
    GDLib
)

# === Python module target ===
pybind11_add_module(autodiff
    src/notebooks/bindings.cpp
    ${AUTODIFF_SOURCES}  # only autodiff sources needed for Python module
)

target_include_directories(autodiff
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set output directory of the Python module to src/notebooks/
set_target_properties(autodiff PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/notebooks
)

# === Summary messages ===
message(STATUS "Autodiff sources found: ${AUTODIFF_SOURCES}")
message(STATUS "Library sources found: ${LIB_SOURCES}")
message(STATUS "Python module will be built as: autodiff")
message(STATUS "Module output directory: ${CMAKE_CURRENT_SOURCE_DIR}/src/notebooks")

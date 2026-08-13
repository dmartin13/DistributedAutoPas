cmake_minimum_required(VERSION 3.16)

project(distributed_autopas LANGUAGES CXX)

find_package(MPI REQUIRED)

# DistributedAutoPas can either be built standalone or as part of a larger
# CMake build that has already configured AutoPas.
if(NOT TARGET autopas)
    get_filename_component(_dap_workspace_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
    set(
        AUTOPAS_SOURCE_DIR
        "${_dap_workspace_dir}/AutoPas"
        CACHE PATH
        "Path to the AutoPas source directory"
    )

    if(NOT EXISTS "${AUTOPAS_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "AutoPas was not found at '${AUTOPAS_SOURCE_DIR}'. "
            "Please set AUTOPAS_SOURCE_DIR to your AutoPas checkout."
        )
    endif()

    # DistributedAutoPas' own standalone build does not need the AutoPas
    # examples/tests. Energy measurements are disabled for the prototype so
    # that PMT/RAPL does not require access to energy_uj.
    set(AUTOPAS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(AUTOPAS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(AUTOPAS_ENABLE_ENERGY_MEASUREMENTS OFF CACHE BOOL "" FORCE)

    add_subdirectory(
        "${AUTOPAS_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}/autopas"
    )
endif()

add_library(distributed_autopas INTERFACE)

target_include_directories(
    distributed_autopas
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(
    distributed_autopas
    INTERFACE
        autopas
        MPI::MPI_CXX
)

target_compile_features(distributed_autopas INTERFACE cxx_std_20)

# Build the local minimal test only when requested. It is ON for a standalone
# DistributedAutoPas build and explicitly disabled by DistributedAutoPasExample.
set(_dap_build_examples_default OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_dap_build_examples_default ON)
endif()
option(
    DISTRIBUTED_AUTOPAS_BUILD_EXAMPLES
    "Build DistributedAutoPas examples"
    ${_dap_build_examples_default}
)

if(DISTRIBUTED_AUTOPAS_BUILD_EXAMPLES)
    add_executable(minimal_mpi examples/minimal_mpi.cpp)
    target_link_libraries(minimal_mpi PRIVATE distributed_autopas)
endif()

find_program(CLANG_FORMAT_EXE clang-format-14)

if(CLANG_FORMAT_EXE)
    file(
        GLOB_RECURSE CLANG_FORMAT_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
    )

    # Use a project-specific target name so this is safe when AutoPas and
    # DistributedAutoPas live in the same superbuild.
    add_custom_target(
        distributed_autopas_clangformat
        COMMAND ${CLANG_FORMAT_EXE} -i ${CLANG_FORMAT_FILES}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Running clang-format for DistributedAutoPas"
    )
else()
    message(STATUS "clang-format-14 not found; formatting target is disabled")
endif()

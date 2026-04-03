cmake_minimum_required(VERSION 3.20)
project(mcfcg LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MCFCG_USE_CUOPT "Enable cuOpt LP solver backend" OFF)
option(MCFCG_USE_MOSEK "Enable MOSEK LP solver backend" OFF)

# HiGHS — always available as default fallback
include(FetchContent)
FetchContent_Declare(
    highs
    GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
    GIT_TAG v1.8.0
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(HIGHS_BUILD_CXX ON CACHE BOOL "" FORCE)
set(BUILD_TESTING_ORIG ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highs)
set(BUILD_TESTING ${BUILD_TESTING_ORIG} CACHE BOOL "" FORCE)

# mcfcg library
add_library(mcfcg
    src/io/commalab_reader.cpp
    src/io/commalab_writer.cpp
    src/io/tntp_reader.cpp
    src/lp/highs_solver.cpp
    src/cg/path_cg.cpp
    src/cg/tree_cg.cpp
)
target_include_directories(mcfcg PUBLIC include)
find_package(ZLIB REQUIRED)
target_link_libraries(mcfcg PUBLIC highs ZLIB::ZLIB)

if(MCFCG_USE_CUOPT)
    target_compile_definitions(mcfcg PUBLIC MCFCG_USE_CUOPT)
    # TODO: find and link cuOpt
endif()

if(MCFCG_USE_MOSEK)
    target_compile_definitions(mcfcg PUBLIC MCFCG_USE_MOSEK)
    # TODO: find and link MOSEK
endif()

# CLI
add_executable(mcfcg_cli src/main.cpp)
target_link_libraries(mcfcg_cli PRIVATE mcfcg)

# Instance cleaning tool
add_executable(mcfcg_clean src/clean.cpp)
target_link_libraries(mcfcg_clean PRIVATE mcfcg)

# Tests
enable_testing()
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(mcfcg_tests
    test/smoke_test.cpp
    test/static_map_test.cpp
    test/static_digraph_test.cpp
    test/d_ary_heap_test.cpp
    test/dijkstra_test.cpp
    test/instance_test.cpp
    test/lp_solver_test.cpp
    test/path_cg_test.cpp
    test/tree_cg_test.cpp
    test/reduced_cost_test.cpp
)
target_link_libraries(mcfcg_tests PRIVATE mcfcg GTest::gtest_main)
target_include_directories(mcfcg_tests PRIVATE include)

include(GoogleTest)
gtest_discover_tests(mcfcg_tests)

add_executable(mcfcg_integration_tests
    test/integration_test.cpp
)
target_link_libraries(mcfcg_integration_tests PRIVATE mcfcg GTest::gtest_main)
target_include_directories(mcfcg_integration_tests PRIVATE include)
target_compile_definitions(mcfcg_integration_tests PRIVATE
    MCFCG_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
gtest_discover_tests(mcfcg_integration_tests)

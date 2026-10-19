cmake_minimum_required(VERSION 3.20)
project(mcfcg LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MCFCG_USE_CUOPT "Enable cuOpt LP solver backend" OFF)
option(MCFCG_CUOPT_DELTA_API
       "Use cuOpt's incremental delta C API (needs the spoorendonk/cuopt fork's cuopt_c_delta.h). ON by default; set OFF only for stock non-fork cuOpt, which falls back to the rebuild-from-scratch path — a serious performance degradation."
       ON)
option(MCFCG_USE_COPT "Enable COPT LP solver backend" OFF)
option(MCFCG_USE_MOSEK "Enable MOSEK LP solver backend" OFF)
option(MCFCG_NATIVE_ARCH "Build with -march=native (host CPU tuning, non-portable binary)" ON)
option(MCFCG_INSTALL_GIT_HOOKS "Point this checkout's core.hooksPath at the tracked .githooks/" ON)

# The git hooks live in .githooks/ under version control, so a fresh clone gets
# them for free -- but git only runs hooks from core.hooksPath, which is
# per-checkout config and cannot be tracked. Configure sets it, since
# `cmake -B build` is the first thing anyone runs here. Idempotent, and it never
# overwrites a value someone else set deliberately: only an unset hooksPath, or
# one already pointing at .githooks, is touched.
# EXISTS, not IS_DIRECTORY: in a linked worktree .git is a file.
if(MCFCG_INSTALL_GIT_HOOKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" config --get core.hooksPath
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE MCFCG_HOOKS_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(NOT MCFCG_HOOKS_PATH)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" config core.hooksPath .githooks
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
            message(STATUS "git hooks: core.hooksPath -> .githooks")
        elseif(NOT MCFCG_HOOKS_PATH STREQUAL ".githooks")
            message(WARNING
                "git hooks: core.hooksPath is '${MCFCG_HOOKS_PATH}', not '.githooks'. "
                "The tracked hooks under .githooks/ are NOT running. "
                "Fix with: git config core.hooksPath .githooks")
        endif()
    endif()
endif()

# ccache when the machine has it. The pre-push local CI builds in a fresh
# worktree every push, so without a compile cache each push recompiles HiGHS
# from scratch. Set before HiGHS is fetched: the launcher is read when a target
# is created, and HiGHS' targets are the recompile worth caching. A user-level
# ccache `base_dir` makes the worktree's paths hit; path-valued -D defines
# (MCFCG_SOURCE_DIR) are not rewritten, so keep those off widely-used targets.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# HiGHS — always available as default fallback
include(FetchContent)
# HiPO (HiGHS interior-point) discards a near-optimal solution as an "internal
# error" when its optional IPX refinement step fails on ill-conditioned masters
# (wide objective range from slack penalty costs). Patched locally; needs
# upstreaming — see gh issue #36. Idempotent: skip if already applied.
set(HIGHS_HIPO_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/highs-hipo-refine-status.patch")
FetchContent_Declare(
    highs
    GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
    GIT_TAG v1.15.1
    PATCH_COMMAND bash -c "git apply --reverse --check '${HIGHS_HIPO_PATCH}' 2>/dev/null || git apply '${HIGHS_HIPO_PATCH}'"
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(HIGHS_BUILD_CXX ON CACHE BOOL "" FORCE)
set(HIPO ON CACHE BOOL "" FORCE)
# HiPO's numerical backend (SuiteSparse AMD, METIS, SPARSEPAK/RCM, BLAS) ships in
# a separate "extras" library. HiGHS defaults BUILD_SHARED_EXTRAS_LIB=ON, which
# builds it as libhighs_extras.so and #defines HIGHS_SHARED_EXTRAS_LIBRARY so the
# feature set is resolved by loading that .so at runtime. Because we link HiGHS
# statically (BUILD_SHARED_LIBS OFF) and never ship/find that .so, the extras go
# unresolved at runtime: HiGHS prints "features unavailable: amd, blas, metis,
# rcm" and SILENTLY falls back to dual simplex while the solver is still nominally
# "hipo". Forcing the extras static compiles them into libhighs so HiPO is
# actually available in the CLI/tests. Verify with `mcfcg_cli --verbose-solver`:
# the HiGHS log must show HiPO/IPX, not "Using dual simplex solver".
set(BUILD_SHARED_EXTRAS_LIB OFF CACHE BOOL "" FORCE)
# HiPO needs a system BLAS (gh #49), and it is found here, before HiGHS:
#  * Without one, HiGHS' own configure stops deep in its FetchContent tree with
#    a bare "Could NOT find BLAS"; this says what to install instead.
#  * The BLAS::BLAS target made here is the one HiGHS links (FindBLAS makes the
#    target only once), and the BLAS_LIBRARIES set here makes HiGHS'
#    FindHipoDeps skip find_package(OpenBLAS CONFIG). Ubuntu's
#    OpenBLASConfig.cmake (libopenblas-dev, multiarch) resolves its root one
#    level too high and names /usr/lib/lib/libopenblas.so, which broke the link.
#  * OpenBLAS is tried first, as HiGHS does: HiGHS defines HIPO_USES_OPENBLAS
#    (and calls openblas_set_num_threads) whenever an OpenBLAS search succeeds,
#    so linking another BLAS while OpenBLAS is also installed (FindBLAS's
#    vendor-less order puts MKL and FlexiBLAS first) would fail at link time.
#    An explicit -DBLA_VENDOR is left to decide.
# macOS uses Accelerate, which HiGHS finds itself.
if(NOT APPLE)
    if(NOT BLA_VENDOR)
        set(BLA_VENDOR OpenBLAS)
        find_package(BLAS QUIET)
        unset(BLA_VENDOR)
    endif()
    if(NOT BLAS_FOUND)
        find_package(BLAS QUIET)
    endif()
    if(NOT BLAS_FOUND)
        message(FATAL_ERROR
            "No BLAS found. HiGHS' interior-point solver (HiPO) needs one: "
            "install OpenBLAS (Debian/Ubuntu: `sudo apt install libopenblas-dev`) "
            "and re-run cmake.")
    endif()
    message(STATUS "BLAS for HiPO: ${BLAS_LIBRARIES}")
endif()
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
    src/source/source_lp.cpp
)
target_include_directories(mcfcg PUBLIC include)
find_package(ZLIB REQUIRED)
target_link_libraries(mcfcg PUBLIC highs ZLIB::ZLIB)

# PUBLIC so the test/CLI binaries (which compile mcfcg's header-only
# templates as part of their own TUs) also benefit from native tuning.
# Disable with -DMCFCG_NATIVE_ARCH=OFF for portable binaries.
if(MCFCG_NATIVE_ARCH AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_options(mcfcg PUBLIC -march=native)
endif()

if(MCFCG_USE_CUOPT)
    target_compile_definitions(mcfcg PUBLIC MCFCG_USE_CUOPT)
    # CUOPT_ROOT accepts either an install tree ({root}/include + {root}/lib[64])
    # or a source-tree checkout of spoorendonk/cuopt ({root}/cpp/include +
    # {root}/cpp/build). CONDA_PREFIX covers the conda-installed case (e.g.
    # a cuopt_dev env that already has runtime deps like rmm, cublas, etc.).
    find_path(CUOPT_INCLUDE_DIR cuopt/linear_programming/cuopt_c.h
              HINTS ENV CUOPT_ROOT ENV CONDA_PREFIX
              PATH_SUFFIXES include cpp/include)
    find_library(CUOPT_LIBRARY cuopt
                 HINTS ENV CUOPT_ROOT ENV CONDA_PREFIX
                 PATH_SUFFIXES lib lib64 cpp/build)
    if(NOT CUOPT_INCLUDE_DIR OR NOT CUOPT_LIBRARY)
        message(FATAL_ERROR "cuOpt not found. Set CUOPT_ROOT (install prefix or spoorendonk/cuopt source checkout), activate a conda env with cuopt installed, or set CMAKE_PREFIX_PATH.")
    endif()
    target_sources(mcfcg PRIVATE src/lp/cuopt_solver.cpp)
    # SYSTEM so clang-tidy (the tidy-fix target) never analyzes/rewrites the
    # vendor SDK headers — a PRIVATE include let modernize-* fix-its corrupt
    # an out-of-repo header.
    target_include_directories(mcfcg SYSTEM PRIVATE ${CUOPT_INCLUDE_DIR})
    target_link_libraries(mcfcg PRIVATE ${CUOPT_LIBRARY})
    # Embed the cuOpt library's directory as a build RPATH so the executables
    # (defined below) resolve libcuopt at load time without the caller setting
    # LD_LIBRARY_PATH. libcuopt itself carries an RPATH to its own deps
    # (librmm / rapids_logger), so this single entry is enough. The path is
    # derived from the configure-time CUOPT_LIBRARY — nothing hardcoded here.
    get_filename_component(_cuopt_libdir "${CUOPT_LIBRARY}" DIRECTORY)
    list(APPEND CMAKE_BUILD_RPATH "${_cuopt_libdir}")
    # Emit DT_RPATH (searched BEFORE LD_LIBRARY_PATH) rather than the linker's
    # default DT_RUNPATH (searched AFTER): a stock cuOpt on the caller's
    # LD_LIBRARY_PATH (e.g. /usr/local/cuopt/lib64) lacks the delta symbols, so
    # it must not win over the fork build pinned here.
    add_link_options(-Wl,--disable-new-dtags)
    # The cuOpt backend defaults to the incremental delta C API: the CG master
    # mutates one persistent LP handle (add/delete rows & columns, resolve) every
    # iteration. It needs the fork that exports cuopt_c_delta.h (spoorendonk/cuopt,
    # epic #20). For stock (non-fork) cuOpt, set -DMCFCG_CUOPT_DELTA_API=OFF to
    # fall back to the rebuild-from-scratch path.
    if(MCFCG_CUOPT_DELTA_API)
        if(NOT EXISTS "${CUOPT_INCLUDE_DIR}/cuopt/linear_programming/cuopt_c_delta.h")
            message(FATAL_ERROR
                "MCFCG_CUOPT_DELTA_API is ON (default) but cuopt_c_delta.h was not found "
                "under ${CUOPT_INCLUDE_DIR} — this looks like a stock (non-fork) cuOpt. "
                "Build the spoorendonk/cuopt fork (delta-api branch, see epic #20), or "
                "reconfigure with -DMCFCG_CUOPT_DELTA_API=OFF to use the rebuild path. "
                "NOTE: the rebuild path recreates the entire LP every CG iteration — a "
                "SERIOUS performance degradation; the delta API is strongly recommended.")
        endif()
        target_compile_definitions(mcfcg PRIVATE MCFCG_CUOPT_DELTA_API)
    else()
        message(WARNING
            "MCFCG_CUOPT_DELTA_API=OFF: using the cuOpt rebuild-from-scratch path, which "
            "recreates the entire LP every CG iteration — SERIOUS performance degradation. "
            "Use only for stock (non-fork) cuOpt; prefer the spoorendonk/cuopt fork with the "
            "delta API left ON.")
    endif()
endif()

if(MCFCG_USE_COPT)
    target_compile_definitions(mcfcg PUBLIC MCFCG_USE_COPT)
    find_path(COPT_INCLUDE_DIR copt.h
              HINTS ENV COPT_HOME
              PATH_SUFFIXES include)
    find_library(COPT_LIBRARY copt
                 HINTS ENV COPT_HOME
                 PATH_SUFFIXES lib lib64)
    if(NOT COPT_INCLUDE_DIR OR NOT COPT_LIBRARY)
        message(FATAL_ERROR "COPT not found. Set COPT_HOME.")
    endif()
    target_sources(mcfcg PRIVATE src/lp/copt_solver.cpp)
    target_include_directories(mcfcg SYSTEM PRIVATE ${COPT_INCLUDE_DIR})
    target_link_libraries(mcfcg PRIVATE ${COPT_LIBRARY})
endif()

if(MCFCG_USE_MOSEK)
    target_compile_definitions(mcfcg PUBLIC MCFCG_USE_MOSEK)
    find_path(MOSEK_INCLUDE_DIR mosek.h
              HINTS ENV MOSEK_HOME
              PATH_SUFFIXES h)
    find_library(MOSEK_LIBRARY mosek64
                 HINTS ENV MOSEK_HOME
                 PATH_SUFFIXES bin)
    if(NOT MOSEK_INCLUDE_DIR OR NOT MOSEK_LIBRARY)
        message(FATAL_ERROR "MOSEK not found. Set MOSEK_HOME (e.g. /opt/mosek/11.0/tools/platform/linux64x86).")
    endif()
    target_sources(mcfcg PRIVATE src/lp/mosek_solver.cpp)
    target_include_directories(mcfcg SYSTEM PRIVATE ${MOSEK_INCLUDE_DIR})
    target_link_libraries(mcfcg PRIVATE ${MOSEK_LIBRARY})
    # Embed libmosek64's directory as a build RPATH so the executables resolve
    # it at load time without the caller setting LD_LIBRARY_PATH. Derived from
    # the configure-time MOSEK_LIBRARY — nothing hardcoded here.
    get_filename_component(_mosek_libdir "${MOSEK_LIBRARY}" DIRECTORY)
    list(APPEND CMAKE_BUILD_RPATH "${_mosek_libdir}")
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
    test/source_lp_test.cpp
    test/thread_pool_test.cpp
)
target_link_libraries(mcfcg_tests PRIVATE mcfcg GTest::gtest_main)
target_include_directories(mcfcg_tests PRIVATE include)

include(GoogleTest)
gtest_discover_tests(mcfcg_tests)

# CLI smoke tests (no source file — just invoke the binary)
add_test(NAME CLI.HelpLong COMMAND mcfcg_cli --help)
set_tests_properties(CLI.HelpLong PROPERTIES PASS_REGULAR_EXPRESSION "Usage:")
add_test(NAME CLI.HelpShort COMMAND mcfcg_cli -h)
set_tests_properties(CLI.HelpShort PROPERTIES PASS_REGULAR_EXPRESSION "Usage:")
add_test(NAME CLI.NoArgs COMMAND mcfcg_cli)
set_tests_properties(CLI.NoArgs PROPERTIES WILL_FAIL true)
add_test(NAME CLI.UnknownFlag COMMAND mcfcg_cli dummy.txt --bogus val)
set_tests_properties(CLI.UnknownFlag PROPERTIES
    PASS_REGULAR_EXPRESSION "Unknown option")

# Python harness tests (stdlib unittest, no new dependency). scripts/ carries the
# provenance of every committed result, and the teardown path these cover already
# cost one benchmark cell its memory measurement SILENTLY — a blank column, no
# error. The -p pattern matches the C++ `<module>_test` convention; unittest's
# default of `test*.py` would discover nothing.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME Python.Scripts
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_SOURCE_DIR}/test/python -p "*_test.py")
    # Python >= 3.12 exits 5 on an empty discovery, but 3.11 and older exit 0 --
    # a renamed file or a broken -p would pass having run nothing. Assert the
    # count too. TIMEOUT bounds the teardown paths, which wait on flush windows.
    set_tests_properties(Python.Scripts PROPERTIES
        TIMEOUT 300
        FAIL_REGULAR_EXPRESSION "Ran 0 tests")
else()
    message(STATUS "Python3 interpreter not found — skipping Python.Scripts tests")
endif()

add_executable(mcfcg_integration_tests
    test/integration_test.cpp
)
target_link_libraries(mcfcg_integration_tests PRIVATE mcfcg GTest::gtest_main)
target_include_directories(mcfcg_integration_tests PRIVATE include)
target_compile_definitions(mcfcg_integration_tests PRIVATE
    MCFCG_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
gtest_discover_tests(mcfcg_integration_tests)

# Lint gate: clang_format and clang_tidy:<path> ctest tests, plus tidy-fix.
include(cmake/Lint.cmake)

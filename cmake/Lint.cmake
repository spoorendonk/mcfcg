# The lint gate: clang-format and clang-tidy as ctest tests labelled `lint`.
#
#   clang_format          -- `clang-format --dry-run -Werror` over every tracked
#                            C++ file (cmake/ClangFormatCheck.cmake).
#   clang_tidy:<path>     -- one test per first-party translation unit, path from
#                            the repository root, whose command is clang-tidy
#                            itself. `.clang-tidy` sets WarningsAsErrors '*', so
#                            any finding fails the test.
#
# The `clang_tidy:` naming and the bare clang-tidy command are what the pre-push
# hook keys on: it runs the suite without those tests and then tidies only what
# the push changed -- a changed source's own unit, a changed header through a
# one-line stub unit -- using the ninja dependency log (hence `-G Ninja` in the
# `build` block) and a cache. So keep the command clang-tidy with no wrapper, and
# set no test property outside LABELS/TIMEOUT/PROCESSORS/WORKING_DIRECTORY.
#
# `ctest -L lint` runs the whole gate; `ctest -LE lint` skips it.
#
# Both tidy flags below are necessary, not decoration:
#   * HeaderFilterRegex from an auto-discovered .clang-tidy is NOT applied.
#     Without --config-file, `clang-tidy -p build src/cg/tree_cg.cpp` reports
#     2 diagnostics where the same run with it reports 481 -- i.e. every
#     header, which is where nearly all of this codebase lives, goes
#     unchecked.
#   * A bare '.*' header filter is the wrong value even so: it matches
#     build/_deps/highs-src/**, where src/lp/highs_solver.cpp alone yields
#     351 diagnostics out of Highs.h. Anchor it at the source dir instead.
#
# Nothing has to be compiled first: HiGHS' generated headers (HConfig.h via
# configure_file, highs_export.h via generate_export_header) and the
# FetchContent sources all exist after `cmake -B build`, which is what lets CI
# lint without building HiGHS.
#
# TOOL VERSIONS ARE PART OF THE CONTRACT: both tools change their output between
# major versions. A different major than MCFCG_CLANG_TOOLS_VERSION is a
# configure-time warning, as is a missing tool; MCFCG_REQUIRE_LINT=ON (CI)
# makes both fatal so the gate can never silently disappear.

set(MCFCG_CLANG_TOOLS_VERSION "21"
    CACHE STRING "Expected clang-format / clang-tidy major version for the lint gate")
option(MCFCG_REQUIRE_LINT
       "Fail configuration when clang-format/clang-tidy are missing or the wrong major version" OFF)

function(_mcfcg_lint_complain)
    string(JOIN "" _msg ${ARGN})
    if(MCFCG_REQUIRE_LINT)
        message(FATAL_ERROR "lint gate: ${_msg} (MCFCG_REQUIRE_LINT is ON)")
    else()
        message(WARNING "lint gate: ${_msg}. Configure with -DMCFCG_REQUIRE_LINT=ON "
                        "to make this fatal.")
    endif()
endfunction()

# Found means usable: present, runnable, and of the expected major version.
# A mismatched major still registers the tests (with a warning) -- the findings
# may differ from CI's, but a gate that disappears is worse.
function(_mcfcg_lint_find var label)
    find_program(${var} NAMES ${label}-${MCFCG_CLANG_TOOLS_VERSION} ${label} NAMES_PER_DIR
                 DOC "${label} binary used by the lint gate")
    if(NOT ${var})
        _mcfcg_lint_complain("${label} not found -- its tests are not registered")
        return()
    endif()
    execute_process(COMMAND "${${var}}" --version
                    OUTPUT_VARIABLE _v ERROR_QUIET RESULT_VARIABLE _rc)
    string(REGEX MATCH "version ([0-9]+)\\." _m "${_v}")
    if(NOT _rc EQUAL 0 OR _m STREQUAL "")
        _mcfcg_lint_complain("${label} at ${${var}} did not report a version")
    elseif(NOT CMAKE_MATCH_1 STREQUAL MCFCG_CLANG_TOOLS_VERSION)
        _mcfcg_lint_complain("${label} is major ${CMAKE_MATCH_1}, the gate expects "
                             "${MCFCG_CLANG_TOOLS_VERSION}; its findings will differ from CI's")
    endif()
endfunction()

_mcfcg_lint_find(CLANG_FORMAT_EXE clang-format)
_mcfcg_lint_find(CLANG_TIDY_EXE clang-tidy)

# The check lists tracked files, so it needs a git checkout; a source tarball
# gets no clang_format test rather than one that always fails. EXISTS, not
# IS_DIRECTORY: in a linked worktree .git is a file.
if(CLANG_FORMAT_EXE AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    message(STATUS "lint gate: not a git checkout -- no clang_format test")
elseif(CLANG_FORMAT_EXE)
    add_test(NAME clang_format
        COMMAND "${CMAKE_COMMAND}"
                "-DCLANG_FORMAT_EXE=${CLANG_FORMAT_EXE}"
                "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ClangFormatCheck.cmake")
    set_tests_properties(clang_format PROPERTIES LABELS lint)
endif()

if(NOT CLANG_TIDY_EXE)
    return()
endif()

# Source list derived from the targets themselves, never a glob: when a
# backend option is OFF its .cpp is not in the target and therefore not in
# compile_commands.json, and clang-tidy hard-errors on a file missing from
# the compilation database. Reading SOURCES keeps the lint set equal to the
# compile-DB set by construction, and a new .cpp is enrolled automatically.
set(MCFCG_TIDY_TARGETS mcfcg mcfcg_cli mcfcg_clean mcfcg_tests mcfcg_integration_tests)
set(MCFCG_TIDY_SOURCES "")
foreach(_tgt IN LISTS MCFCG_TIDY_TARGETS)
    get_target_property(_tgt_dir ${_tgt} SOURCE_DIR)
    get_target_property(_tgt_srcs ${_tgt} SOURCES)
    foreach(_src IN LISTS _tgt_srcs)
        if(NOT _src MATCHES "\\.(cpp|cc|cxx)$")
            continue()  # headers are covered through --header-filter
        endif()
        if(IS_ABSOLUTE "${_src}")
            list(APPEND MCFCG_TIDY_SOURCES "${_src}")
        else()
            list(APPEND MCFCG_TIDY_SOURCES "${_tgt_dir}/${_src}")
        endif()
    endforeach()
endforeach()
list(REMOVE_DUPLICATES MCFCG_TIDY_SOURCES)
list(LENGTH MCFCG_TIDY_SOURCES MCFCG_TIDY_COUNT)

# Escape every character that is not [A-Za-z0-9_/-] so a checkout path
# containing '.', '+' or '(' cannot alter the regex.
string(REGEX REPLACE "([^A-Za-z0-9_/-])" "\\\\\\1"
       _mcfcg_root_re "${CMAKE_CURRENT_SOURCE_DIR}")
set(MCFCG_TIDY_HEADER_FILTER "^${_mcfcg_root_re}/(include|src|test)/")

foreach(_src IN LISTS MCFCG_TIDY_SOURCES)
    file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_src}")
    add_test(NAME "clang_tidy:${_rel}"
        COMMAND "${CLANG_TIDY_EXE}"
                -p "${CMAKE_BINARY_DIR}"
                --quiet
                "--config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy"
                "--header-filter=${MCFCG_TIDY_HEADER_FILTER}"
                "${_src}")
    # A heavy TU (cg_loop.h through tree_cg.cpp) takes ~30s; leave headroom
    # for a loaded laptop.
    set_tests_properties("clang_tidy:${_rel}" PROPERTIES LABELS lint TIMEOUT 600)
endforeach()

# tidy-fix applies fix-its. Two rails:
#   1. Refuse to run on a dirty worktree. add_custom_target aborts on the
#      first non-zero COMMAND, so `git diff --quiet` IS the gate: it
#      guarantees the whole effect is reviewable as one `git diff` and
#      revertible with `git checkout .`.
#   2. Serial, one file at a time -- NOT run-clang-tidy -fix, and never a
#      parallel loop. Concurrent `clang-tidy --fix` processes would rewrite
#      the same shared header in include/mcfcg simultaneously. run-clang-tidy
#      avoids that by batching through clang-apply-replacements, but it was
#      measured here silently applying only a fraction of the fixes it had
#      just reported, which is worse than being slow.
# The anchored --header-filter is what keeps fix-its inside
# include|src|test; the SYSTEM include dirs for the vendor SDKs are the
# independent second mechanism.
set(MCFCG_TIDY_FIX_CHECKS ""
    CACHE STRING "Restrict tidy-fix to these checks; empty uses .clang-tidy.")
set(_tidy_fix_args "")
if(MCFCG_TIDY_FIX_CHECKS)
    set(_tidy_fix_args "--checks=${MCFCG_TIDY_FIX_CHECKS}")
endif()
add_custom_target(tidy-fix
    COMMAND "${CMAKE_COMMAND}" -E echo
            "tidy-fix: requires a clean worktree (fixes are reviewed via git diff)"
    COMMAND "${CMAKE_COMMAND}"
            -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TidyFixGuard.cmake"
    COMMAND "${CMAKE_COMMAND}"
            -DCLANG_TIDY_EXE=${CLANG_TIDY_EXE}
            -DBUILD_DIR=${CMAKE_BINARY_DIR}
            -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            "-DHEADER_FILTER=${MCFCG_TIDY_HEADER_FILTER}"
            "-DEXTRA_ARGS=${_tidy_fix_args}"
            "-DSOURCES=${MCFCG_TIDY_SOURCES}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TidyFix.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "clang-tidy --fix over ${MCFCG_TIDY_COUNT} first-party TUs (serial)"
    USES_TERMINAL VERBATIM)

message(STATUS "clang-tidy: ${CLANG_TIDY_EXE} -- ${MCFCG_TIDY_COUNT} clang_tidy: tests")

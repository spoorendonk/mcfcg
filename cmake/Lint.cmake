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
# TOOL VERSIONS ARE PART OF THE CONTRACT: both tools change their output and
# their check set between versions, so the gate would pass on one machine and
# fail on another. `.clang-format-version` and `.clang-tidy-version` hold the
# group's one toolchain version, 22.1.8 (changed for every repo together,
# through the hub), and the WHOLE version is compared, not the major. The tools
# are the wheels pyproject.toml pins (`uv sync`), in `.venv/bin`, searched before
# anything else, so PATH order does not matter. Only this tree's: the push's
# worktree has none, and there the hook's PATH already holds the pushing tree's,
# then the main checkout's, which a hint here would overrule. A linked worktree
# configured by hand needs its own `uv sync`. NO_CACHE, so a venv created after
# the first configure is picked up; -DCLANG_TIDY_EXE=... still overrides when
# it is the pin, and a preset or cached path that is not is dropped.
#
# A missing tool, a missing pin or any other version is fatal under
# MCFCG_REQUIRE_LINT, which the `build` block (the push gate) and the weekly CI
# set, naming the found and wanted versions, so a push cannot land unlinted.
# Without it (a hand-configured build) it is a warning that drops that tool's
# tests, so a machine without the tools still builds.

option(MCFCG_REQUIRE_LINT
       "Fail the configure unless clang-format and clang-tidy are exactly the pinned versions"
       OFF)

# A tool the gate cannot use: fatal under MCFCG_REQUIRE_LINT, else a warning.
# An optional fourth argument replaces the default advice (`uv sync`).
function(_mcfcg_lint_unusable tool wanted problem)
    set(_fix "Run `uv sync` for the pinned wheels in .venv/bin (pyproject.toml)")
    if(ARGC GREATER 3)
        set(_fix "${ARGV3}")
    endif()
    if(MCFCG_REQUIRE_LINT)
        message(FATAL_ERROR "lint gate: ${problem}, wanted ${tool} ${wanted}. ${_fix}.")
    endif()
    message(WARNING "lint gate: ${problem}, wanted ${tool} ${wanted}: its tests are not "
                    "registered. ${_fix}; -DMCFCG_REQUIRE_LINT=ON, as the push gate's "
                    "build block has, makes this fatal.")
endfunction()

# Sets <out> to the whole version token after "version" in <exe>'s banner, or
# "" when it reports none.
function(_mcfcg_lint_version out exe)
    execute_process(COMMAND "${exe}" --version OUTPUT_VARIABLE _banner ERROR_QUIET)
    set(_found "")
    if(_banner MATCHES "version ([^ \t\r\n]+)")
        set(_found "${CMAKE_MATCH_1}")
    endif()
    set(${out} "${_found}" PARENT_SCOPE)
endfunction()

# Sets <var> to the path of <tool> when it is exactly the version
# .<tool>-version pins, else clears it. The whole token after "version", so a
# suffix joined to the number ("22.1.8-rc1") is not the pin. A pin file missing
# or empty counts as unusable: nothing to compare against must not pass.
function(_mcfcg_lint_find var tool)
    set(_file "${CMAKE_CURRENT_SOURCE_DIR}/.${tool}-version")
    set(_wanted "")
    if(EXISTS "${_file}")
        file(STRINGS "${_file}" _wanted LIMIT_COUNT 1)
        string(STRIP "${_wanted}" _wanted)
    endif()
    if(_wanted STREQUAL "")
        _mcfcg_lint_unusable(${tool} "the pinned version" ".${tool}-version holds no version"
                             "Restore .${tool}-version from git (the group pins 22.1.8)")
        set(${var} "" PARENT_SCOPE)
        return()
    endif()
    # find_program does not search when <var> is already set, and the old
    # Lint.cmake cached it, so a build dir from before the pin would keep its
    # old tool past `uv sync`. A preset path (-D or a stale cache entry) that is
    # the pin wins; one that is not is dropped and the search runs.
    if(${var})
        _mcfcg_lint_version(_preset "${${var}}")
        if(NOT _preset STREQUAL _wanted)
            message(STATUS "lint gate: ignoring ${var}=${${var}} (${tool} '${_preset}', "
                           "not ${_wanted}) and searching .venv/bin, then PATH")
            unset(${var} CACHE)
            unset(${var})
        endif()
    endif()
    # Not CMAKE_PREFIX_PATH or CMAKE_PROGRAM_PATH, which CMake would search
    # before the hint.
    find_program(${var} ${tool} HINTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin"
                 NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CACHE)
    if(NOT ${var})
        _mcfcg_lint_unusable(${tool} "${_wanted}" "no ${tool} found")
        set(${var} "" PARENT_SCOPE)
        return()
    endif()
    _mcfcg_lint_version(_found "${${var}}")
    if(NOT _found STREQUAL _wanted)
        if(_found STREQUAL "")
            set(_found "of an unknown version")
        endif()
        _mcfcg_lint_unusable(${tool} "${_wanted}" "${${var}} is ${tool} ${_found}")
        set(${var} "" PARENT_SCOPE)
        return()
    endif()
    set(${var} "${${var}}" PARENT_SCOPE)
endfunction()

_mcfcg_lint_find(CLANG_FORMAT_EXE clang-format)
_mcfcg_lint_find(CLANG_TIDY_EXE clang-tidy)

# The wheel's clang-tidy has no default SDK on macOS, so without a sysroot it
# does not find the C++ standard library there. Linux needs nothing.
set(MCFCG_TIDY_EXTRA_ARGS "")
if(APPLE AND CLANG_TIDY_EXE)
    execute_process(COMMAND xcrun --show-sdk-path
                    OUTPUT_VARIABLE _mcfcg_sdk OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_mcfcg_sdk)
        set(MCFCG_TIDY_EXTRA_ARGS "--extra-arg=-isysroot${_mcfcg_sdk}")
    endif()
endif()

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
                ${MCFCG_TIDY_EXTRA_ARGS}
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
set(_tidy_fix_args ${MCFCG_TIDY_EXTRA_ARGS})
if(MCFCG_TIDY_FIX_CHECKS)
    list(APPEND _tidy_fix_args "--checks=${MCFCG_TIDY_FIX_CHECKS}")
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

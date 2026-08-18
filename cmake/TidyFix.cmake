# Serial clang-tidy --fix driver behind the `tidy-fix` target.
#
# Runs in CMake script mode (cmake -P) so the loop is one process writing at a
# time. That is the point: every template in this project lives in a header
# under include/mcfcg, so N concurrent `clang-tidy --fix` processes would
# rewrite the same file simultaneously -- interleaved, corrupt edits, not a
# race you can retry. run-clang-tidy sidesteps that by batching through
# clang-apply-replacements, but it was measured here reporting a set of
# warnings and then silently applying only a fraction of their fixes, so this
# trades its parallelism for actually applying what it reports.
#
# Expects: CLANG_TIDY_EXE, BUILD_DIR, SOURCE_DIR, HEADER_FILTER, SOURCES
# (a ;-list), and optionally EXTRA_ARGS.

if(NOT CLANG_TIDY_EXE OR NOT BUILD_DIR OR NOT SOURCES)
    message(FATAL_ERROR "TidyFix.cmake: CLANG_TIDY_EXE, BUILD_DIR and SOURCES are required")
endif()

set(_extra "")
if(EXTRA_ARGS)
    set(_extra "${EXTRA_ARGS}")
endif()

list(LENGTH SOURCES _total)
set(_index 0)
set(_failed "")

foreach(_src IN LISTS SOURCES)
    math(EXPR _index "${_index} + 1")
    file(RELATIVE_PATH _rel "${SOURCE_DIR}" "${_src}")
    message(STATUS "[${_index}/${_total}] clang-tidy --fix ${_rel}")
    execute_process(
        COMMAND "${CLANG_TIDY_EXE}"
                -p "${BUILD_DIR}"
                --quiet
                "--config-file=${SOURCE_DIR}/.clang-tidy"
                "--header-filter=${HEADER_FILTER}"
                --fix
                --format-style=file
                ${_extra}
                "${_src}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE _rc)
    # A non-zero exit here means diagnostics in the WarningsAsErrors set
    # survived the fix pass (not every check has a fix-it). Keep going so one
    # unfixable file does not hide the fixes available in the rest, then report.
    if(NOT _rc EQUAL 0)
        list(APPEND _failed "${_rel}")
    endif()
endforeach()

if(_failed)
    list(JOIN _failed "\n  " _failed_text)
    message(STATUS
        "tidy-fix: fixes applied, but these files still have findings with no "
        "fix-it (review them by hand):\n  ${_failed_text}")
endif()

message(STATUS "tidy-fix: done -- review the result with `git diff`")

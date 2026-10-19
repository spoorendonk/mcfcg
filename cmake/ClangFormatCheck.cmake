# `clang-format --dry-run -Werror` over every tracked C++ file, behind the
# `clang_format` ctest test.
#
# Tracked files, not a glob: the same set the pre-commit hook checks, and a glob
# would also pick up untracked scratch files (the pre-push lint runner writes
# `.tidy-stub-<n>.cpp` beside headers).
#
# Expects: CLANG_FORMAT_EXE, SOURCE_DIR.

if(NOT CLANG_FORMAT_EXE OR NOT SOURCE_DIR)
    message(FATAL_ERROR "ClangFormatCheck.cmake: CLANG_FORMAT_EXE and SOURCE_DIR are required")
endif()

find_package(Git QUIET)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "ClangFormatCheck.cmake: git is needed to list the tracked sources")
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" ls-files -- "*.h" "*.hpp" "*.cpp" "*.cc"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE _files
    RESULT_VARIABLE _rc
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _rc EQUAL 0 OR _files STREQUAL "")
    # An empty list would pass having checked nothing.
    message(FATAL_ERROR "ClangFormatCheck.cmake: `git ls-files` listed no sources in ${SOURCE_DIR}")
endif()
string(REPLACE "\n" ";" _files "${_files}")

execute_process(
    COMMAND "${CLANG_FORMAT_EXE}" --dry-run -Werror ${_files}
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR
        "clang-format: files above are unformatted. Fix with:\n"
        "  git ls-files -z '*.h' '*.cpp' | xargs -0 clang-format -i")
endif()

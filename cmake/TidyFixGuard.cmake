# Worktree guard for the `tidy-fix` target.
#
# tidy-fix rewrites source in place, and its whole safety story is "the entire
# effect is one reviewable `git diff`, revertible with `git checkout .`". That
# promise needs two things a bare `git diff --quiet HEAD --` does not give:
#
#   * Untracked files count. `git checkout .` cannot restore a file git has
#     never seen, so an unstaged new header rewritten by a fix-it would be
#     unrecoverable. Check porcelain status over the linted areas instead.
#   * A missing checkout must fail comprehensibly. The published artifact is
#     unpacked from a tarball with no .git, where `git diff` dies with
#     "not a git repository" and tells the user nothing useful.

if(NOT SOURCE_DIR)
    message(FATAL_ERROR "TidyFixGuard.cmake: SOURCE_DIR is required")
endif()

execute_process(
    COMMAND git -C "${SOURCE_DIR}" rev-parse --git-dir
    RESULT_VARIABLE _no_git OUTPUT_QUIET ERROR_QUIET)
if(NOT _no_git EQUAL 0)
    message(FATAL_ERROR
        "tidy-fix refuses to run outside a git checkout: it rewrites source in "
        "place and relies on `git diff` for review and `git checkout .` for "
        "undo. Use the read-only `tidy` target instead, or apply fixes in a "
        "clone.")
endif()

execute_process(
    COMMAND git -C "${SOURCE_DIR}" status --porcelain -- include src test
    OUTPUT_VARIABLE _dirty OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _dirty STREQUAL "")
    message(FATAL_ERROR
        "tidy-fix requires a clean worktree under include/, src/ and test/ -- "
        "fixes are reviewed via `git diff` and undone via `git checkout .`, and "
        "neither works if there are already-modified or untracked files there. "
        "Commit or stash first.\n${_dirty}")
endif()

message(STATUS "tidy-fix: worktree clean, applying fixes")

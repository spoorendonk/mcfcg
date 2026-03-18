# mcfcg

## Quick Reference

```bash
# build
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# test
ctest --test-dir build -j$(nproc)

# format
find src include -name '*.cpp' -o -name '*.h' | xargs clang-format -i
```

## Git

- Never commit directly to `main`. Always feature branches.
- Linear history only (squash-merge).
- Squash-merge is the only allowed merge method.

## Workflow: Plan → Grind

Every task has two phases. Do not skip planning.

### Plan (default)

When given a task, **plan first**: investigate the code, propose an approach,
discuss with the user. Wait for approval before implementing (e.g. "grind", "go", "do it").

### Grind (on approval)

Execute autonomously. Build, test, fix, repeat until green.
Self-review, then fullgate: branch, PR, sync main, push.
Progress lives in files and git — not in your context window.

Only pause and ask a human when:
- A fix requires changing the public API or architecture
- You discover a bug in unrelated code you shouldn't touch
- You're stuck after multiple failed attempts

### Fullgate

Also runs standalone when user says **"fullgate"**:
branch → PR → sync (merge main **into** feature branch) → tests → docs →
push → review → build → test → push fixes → squash-merge → delete branch

## Review Loop

After self-review or PR review, fix all issues and nits, then re-review.
Repeat until no more issues or nits remain.

## Claiming Work (GitHub)

- `gh issue edit <N> --add-label agent-wip` when starting on an issue or PR
- Check for `agent-wip` label before picking up work
- Remove label and close/merge when done

## Teams

For independent sub-tasks, launch a team. Each teammate works in its own
worktree. Lead integrates: merge, resolve conflicts, build/test the result.

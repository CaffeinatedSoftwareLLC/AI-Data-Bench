# Lasso claude-hooks: pattern file

`patterns.yaml` is copied unchanged from
[lasso-security/claude-hooks](https://github.com/lasso-security/claude-hooks) at commit
`8fbfd14c2a2271fe58a95c416b3b31880458c7bc` (2026-01-08), path
`.claude/skills/prompt-injection-defender/patterns.yaml`. MIT licence, copied alongside as `LICENSE`.

The `Lasso claude-hooks` arm in `adapters.py` matches it the way the hook's
`post-tool-defender.py` does: all four categories, `re.IGNORECASE | re.MULTILINE`, invalid patterns skipped,
text under 10 characters skipped, and a warning on any match whatever its severity. Score = 1 on any match, else 0.

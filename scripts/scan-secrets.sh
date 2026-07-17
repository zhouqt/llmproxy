#!/usr/bin/env bash
# Scan the llmproxy project for accidental API key leaks.
#
# Checks two surfaces:
#   1. Working tree — every tracked text file (excluding .git, target,
#      .claude session data, build artifacts, and this script itself).
#   2. Git history — every commit's diff, so a secret that was added
#      and later removed is still caught.
#
# Patterns are deliberately narrow to keep false positives low. Each
# match is printed with file:line so the operator can verify. The proxy
# test key `sk-llmproxy-1234` is whitelisted (used by integration tests
# with wiremock + the live podman container).
#
# Exit status:
#   0 — clean (no matches)
#   1 — at least one potential match (review output)
#
# Dependencies: GNU grep (works on stock Linux/macOS), git.

set -uo pipefail

ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$ROOT"

# grep --exclude-dir / --exclude flags. Glob forms (*.lock) are matched
# against the basename by grep without `wildcard` semantics here; for
# directory globs we list the trailing slash form.
EXCLUDE_DIRS=(
  '.git'
  'target'
  '.claude'
  'coverage'
)
EXCLUDE_FILES=(
  'Cargo.lock'
  '*.lock.bak'
  '*.lcov'
  '*.profraw'
  '*.profdata'
  'scan-secrets.sh'
  'scripts/scan-secrets.sh'
)

# Patterns. grep -E extended regex. Length thresholds trim the
# false-positive tail (any random 8-char word would otherwise match).
PATTERNS=(
  # OpenAI / DeepSeek / generic sk- keys (>= 20 chars after prefix).
  # Long enough that "sk-test-key" type values fall through but real
  # OpenAI/DeepSeek keys (`sk-8dd34df0...`-style, 48 chars) are caught.
  'sk-[A-Za-z0-9_-]{20,}'
  # Anthropic native
  'sk-ant-[A-Za-z0-9_-]{20,}'
  # GitHub classic PAT + OAuth / installation / refresh tokens
  '(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}'
  # GitHub fine-grained PAT
  'github_pat_[A-Za-z0-9_]{20,}'
  # AWS access key ID
  'AKIA[0-9A-Z]{16}'
  # Slack tokens
  'xox[abprs]-[A-Za-z0-9-]{10,}'
)

# Whitelisted substrings — known-safe values used by tests / docs.
# sk-llmproxy-1234: integration-test key (wiremock + live container)
# sk-fakeabc123def456ghi789jkl012: scanner's own documentation example
#   in scripts/hooks/pre-commit's commit message; intentionally a fake
#   key that exercises the `FOUND` path without being a real secret.
WHITELIST_REGEX='sk-llmproxy-1234|sk-fakeabc123def456ghi789jkl012'

matches=0

build_grep_excludes() {
  local args=()
  for d in "${EXCLUDE_DIRS[@]}"; do args+=(--exclude-dir "$d"); done
  for f in "${EXCLUDE_FILES[@]}"; do args+=(--exclude "$f"); done
  printf '%s\n' "${args[@]}"
}

scan_working_tree() {
  echo "==> Scanning working tree at $ROOT"
  local GREP_ARGS
  GREP_ARGS=$(build_grep_excludes)
  # shellcheck disable=SC2086
  set -- $GREP_ARGS

  for p in "${PATTERNS[@]}"; do
    # grep -rE exits 1 on no matches; that's expected for most patterns.
    local out
    if ! out=$(grep -rEn --color=never "$@" "$p" . 2>/dev/null); then
      continue
    fi
    [ -z "$out" ] && continue

    # Filter the test key out. grep -v exits 1 if no line survives;
    # || true keeps the pipeline going.
    local filtered
    filtered=$(printf '%s\n' "$out" | grep -Ev "$WHITELIST_REGEX" || true)
    [ -z "$filtered" ] && continue

    while IFS= read -r line; do
      [ -z "$line" ] && continue
      printf '  WORKTREE  %s\n' "$line"
      matches=$((matches + 1))
    done <<< "$filtered"
  done
}

scan_git_history() {
  # `git log --all -p` shows every commit's full diff. Includes lines
  # from commits that have since been reverted — exactly what we want
  # for a "did we ever commit this?" check. For repos with thousands
  # of commits this is slow; for ours (tens of commits) it's instant.
  echo "==> Scanning git history (git log --all -p)"
  local p
  for p in "${PATTERNS[@]}"; do
    local diff
    diff=$(git log --all -p 2>/dev/null || true)
    [ -z "$diff" ] && continue

    # Capture the lines we care about with a precise filter:
    # - Drop file metadata (---, +++, diff --git, index)
    # - Keep added ('+') and context (' ') lines only; '-' lines are
    #   removals — we want to flag any added secret, even if it was
    #   later removed.
    # - Require the line to match $p directly.
    local filtered
    filtered=$(printf '%s\n' "$diff" \
      | grep -E "^[+ ]" \
      | grep -E "$p" \
      | grep -Ev "$WHITELIST_REGEX" \
      | grep -Ev '^(---|\+\+\+|index |diff --git)' \
      || true)

    [ -z "$filtered" ] && continue

    while IFS= read -r line; do
      [ -z "$line" ] && continue
      printf '  HISTORY   %s\n' "$line"
      matches=$((matches + 1))
    done <<< "$filtered"
  done
}

scan_working_tree
scan_git_history

echo
if [ "$matches" -eq 0 ]; then
  echo "OK: no API key patterns detected."
  exit 0
fi
echo "FOUND: $matches potential matches. Review each line above before publishing."
exit 1

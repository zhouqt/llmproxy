#!/usr/bin/env bash
# Install the git hooks from scripts/hooks/ into .git/hooks/.
# Idempotent — will skip a hook that already exists as a non-symlink
# (so the operator can decide whether to overwrite a manual hook).
#
# Usage:
#   scripts/install-hooks.sh                # install all
#   scripts/install-hooks.sh --uninstall    # remove installed symlinks
#
# Alternative (per-machine, no installer needed):
#   git config core.hooksPath scripts/hooks
# — but note this also affects submodules and is shared across all
# clones of the same repo on this machine. The symlink approach is
# per-clone and stays local to .git/, so it doesn't leak out when
# the repo is copied to a different machine.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="$ROOT/scripts/hooks"
HOOKS_DST="$ROOT/.git/hooks"

if [ ! -d "$HOOKS_SRC" ]; then
  echo "No scripts/hooks/ directory; nothing to install." >&2
  exit 0
fi

mode="install"
case "${1:-}" in
  install|"") mode="install" ;;
  -u|--uninstall) mode="uninstall" ;;
  -h|--help)
    sed -n '2,/^set -euo pipefail/p' "$0" | sed 's/^# \?//'
    exit 0
    ;;
  *) echo "Unknown arg: $1" >&2; exit 2 ;;
esac

if [ "$mode" = "install" ]; then
  installed=0
  skipped=0
  for hook in "$HOOKS_SRC"/*; do
    [ -f "$hook" ] || continue
    name=$(basename "$hook")
    target="$HOOKS_DST/$name"

    # Don't clobber a manually-installed (non-symlink) hook.
    if [ -e "$target" ] && [ ! -L "$target" ]; then
      echo "  SKIP   $name  (existing file in .git/hooks; remove it first to overwrite)"
      skipped=$((skipped + 1))
      continue
    fi

    ln -sfn "$(realpath "$hook")" "$target"
    chmod +x "$hook"
    echo "  LINK   $name  -> $target"
    installed=$((installed + 1))
  done
  echo
  echo "Installed $installed hook(s); skipped $skipped."
else
  # Uninstall: remove symlinks pointing back into scripts/hooks/.
  removed=0
  for hook in "$HOOKS_SRC"/*; do
    [ -f "$hook" ] || continue
    name=$(basename "$hook")
    target="$HOOKS_DST/$name"
    if [ -L "$target" ]; then
      rm "$target"
      echo "  RM     $name"
      removed=$((removed + 1))
    fi
  done
  echo
  echo "Removed $removed symlink(s)."
fi

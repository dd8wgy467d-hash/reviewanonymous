#!/usr/bin/env bash
# Remove generated artefacts.
#
#   bash scripts/clean.sh [--dry-run] [SCOPE ...]
#
# Scopes (default: all four):
#   caches   __pycache__ and wandb directories, stray .pyc files
#   results  everything under results/
#   figures  everything under figures/
#   logs     everything under logs/
# .venv/ and archive/ are never touched.
set -euo pipefail
BASE="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)"

DRY_RUN=0; SCOPES=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    caches|results|figures|logs) SCOPES+=("$arg") ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[[ ${#SCOPES[@]} -eq 0 ]] && SCOPES=(caches results figures logs)
has() { [[ " ${SCOPES[*]} " == *" $1 "* ]]; }
rm_() { echo "  [remove] $1"; [[ $DRY_RUN -eq 0 ]] && rm -rf "$1"; return 0; }

echo "[clean] $BASE: ${SCOPES[*]}$([[ $DRY_RUN -eq 1 ]] && echo ' (dry run)')"
if has caches; then
  while IFS= read -r -d '' t; do rm_ "$t"; done < <(find "$BASE" \
    -path "$BASE/.venv" -prune -o -path "$BASE/archive" -prune -o \
    \( -type d \( -name __pycache__ -o -name wandb \) -o -type f -name '*.pyc' \) -print0)
fi
for d in results figures logs; do
  if has "$d" && [[ -d "$BASE/$d" ]]; then
    while IFS= read -r -d '' t; do rm_ "$t"; done < <(find "$BASE/$d" -mindepth 1 -maxdepth 1 -print0)
  fi
done
echo "[clean] done"

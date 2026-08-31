#!/usr/bin/env bash
# get_usalign.sh — build the US-align binary into tools/USalign.
#
# The binary is platform-specific, so it is gitignored rather than committed;
# 4_scoring/score.py expects it at tools/USalign. Safe to re-run.
#
# US-align: Zhang et al., Nature Methods 19, 1109-1115 (2022).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/tools"
TARGET="$TOOLS/USalign"
SRC_URL="https://zhanggroup.org/US-align/bin/module/USalign.cpp"

if [[ -x "$TARGET" ]]; then
    echo "US-align already present at $TARGET"
    "$TARGET" -h 2>&1 | head -1 || true
    echo "Delete it and re-run to rebuild."
    exit 0
fi

command -v g++ >/dev/null 2>&1 || {
    echo "error: g++ not found." >&2
    echo "  macOS: xcode-select --install" >&2
    echo "  Debian/Ubuntu: sudo apt-get install build-essential" >&2
    exit 1
}

mkdir -p "$TOOLS"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Downloading USalign.cpp ..."
if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$SRC_URL" -o "$WORK/USalign.cpp"
else
    wget -q "$SRC_URL" -O "$WORK/USalign.cpp"
fi

# Sanity-check what came back — a captive portal or an error page would
# otherwise fail confusingly at the compile step.
if [[ ! -s "$WORK/USalign.cpp" ]] || ! head -40 "$WORK/USalign.cpp" | grep -qi "align"; then
    echo "error: downloaded file does not look like USalign.cpp." >&2
    echo "  Fetch it manually from $SRC_URL and compile into $TARGET" >&2
    exit 1
fi

echo "Compiling ..."
# -static is deliberately omitted: it is unsupported on macOS and unnecessary
# for local use.
g++ -O3 -ffast-math -lm -o "$TARGET" "$WORK/USalign.cpp"
chmod +x "$TARGET"

echo "Built $TARGET"
"$TARGET" -h 2>&1 | head -1 || true

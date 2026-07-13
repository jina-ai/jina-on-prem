#!/usr/bin/env bash
# Sync docs/ to the GitHub wiki.
#
# Prerequisite: the wiki must be initialized first via the web UI - open
# https://github.com/jina-ai/jina-on-prem/wiki, click "Create the first page",
# save anything. This is a one-time manual step (no GitHub API exists).
#
# After that, this script clones the wiki repo, copies docs/* into it,
# and pushes.

set -euo pipefail

REPO="${REPO:-jina-ai/jina-on-prem}"
DOCS_DIR="$(cd "$(dirname "$0")/.." && pwd)/docs"
WIKI_URL="git@github.com:${REPO}.wiki.git"
TMP="$(mktemp -d)"

if [[ ! -d "$DOCS_DIR" ]]; then
  echo "error: docs/ not found at $DOCS_DIR" >&2
  exit 1
fi

echo "Cloning $WIKI_URL..."
if ! git clone --quiet "$WIKI_URL" "$TMP/wiki" 2>&1; then
  cat >&2 <<EOF

Clone failed - the wiki is probably not initialized yet.

One-time setup:
  1. Open https://github.com/${REPO}/wiki
  2. Click "Create the first page"
  3. Save any content
  4. Re-run this script
EOF
  rm -rf "$TMP"
  exit 1
fi

echo "Copying docs/..."
cp "$DOCS_DIR"/*.md "$TMP/wiki/"
mkdir -p "$TMP/wiki/images"
cp -r "$DOCS_DIR/images/." "$TMP/wiki/images/"
rm -f "$TMP/wiki/README.md"   # docs/README.md is for the repo, not the wiki

cd "$TMP/wiki"
if [[ -z "$(git status --porcelain)" ]]; then
  echo "Wiki already up to date."
  rm -rf "$TMP"
  exit 0
fi

git add -A
git -c user.email="${GIT_EMAIL:-noreply@github.com}" \
    -c user.name="${GIT_NAME:-jina-on-prem docs sync}" \
    commit --quiet -m "Sync docs/ from main repo"
git push origin HEAD

echo
echo "Pushed wiki. View at https://github.com/${REPO}/wiki"
rm -rf "$TMP"

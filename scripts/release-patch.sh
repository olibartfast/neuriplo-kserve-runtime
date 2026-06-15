#!/usr/bin/env bash
set -euo pipefail

# GitFlow patch release procedure.
# Usage: scripts/release-patch.sh [--dry-run]
#
# Steps:
#  1. Bump VERSION (patch increment) and CHANGELOG.md on a release/X.Y.Z branch
#  2. Merge release branch into master, tag vX.Y.Z
#  3. Merge release branch back into develop
#  4. Delete the release branch, push master + develop + tags

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No files or remotes will be modified"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

require_clean() {
    if ! git diff-index --quiet HEAD --; then
        echo "ERROR: working tree is dirty. Commit or stash changes first."
        exit 1
    fi
}

require_branch() {
    local current
    current="$(git branch --show-current)"
    if [[ "$current" != "$1" ]]; then
        echo "ERROR: must be on branch '$1', currently on '$current'"
        exit 1
    fi
}

# ── Parse current version ──────────────────────────────────────────────
CUR="$(cat VERSION)"
if [[ ! "$CUR" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "ERROR: VERSION must be semver X.Y.Z, got '$CUR'"
    exit 1
fi
MAJOR="${BASH_REMATCH[1]}"
MINOR="${BASH_REMATCH[2]}"
PATCH="${BASH_REMATCH[3]}"
NEXT="${MAJOR}.${MINOR}.$((PATCH + 1))"
TODAY="$(date +%Y-%m-%d)"

echo "Current:  v${CUR}"
echo "Next:     v${NEXT}  ($TODAY)"
echo

# ── Pre-flight ─────────────────────────────────────────────────────────
require_branch develop
require_clean

if [[ "$DRY_RUN" != "true" ]]; then
    if ! git fetch origin develop master --tags 2>/dev/null; then
        echo "WARNING: could not fetch from origin — continuing with local state"
    fi

    LOCAL="$(git rev-parse HEAD)"
    REMOTE="$(git rev-parse origin/develop 2>/dev/null || echo "$LOCAL")"
    if [[ "$LOCAL" != "$REMOTE" ]]; then
        echo "ERROR: local develop is not in sync with origin/develop"
        echo "  local:  $LOCAL"
        echo "  remote: $REMOTE"
        exit 1
    fi
fi

# ── Create release branch ──────────────────────────────────────────────
BRANCH="release/${NEXT}"
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    echo "ERROR: branch '${BRANCH}' already exists locally"
    exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would create branch: $BRANCH"
    echo "[DRY RUN] Would set VERSION → $NEXT"
    echo "[DRY RUN] Would add [${NEXT}] section to CHANGELOG.md"
    echo "[DRY RUN] Would commit Release v${NEXT}"
    echo "[DRY RUN] Would merge $BRANCH → master, tag v${NEXT}"
    echo "[DRY RUN] Would merge $BRANCH → develop"
    echo "[DRY RUN] Would delete local branch: $BRANCH"
    echo "[DRY RUN] Would push master, develop, tags"
    echo
    echo "[DRY RUN] Release v${NEXT} complete (no changes made)"
    exit 0
fi

# ── Execute release ────────────────────────────────────────────────────
git checkout -b "$BRANCH"
echo "Created branch: $BRANCH"

# Bump VERSION
echo "$NEXT" > VERSION
echo "VERSION → $NEXT"

# Bump CHANGELOG.md
CHANGELOG="CHANGELOG.md"
if grep -q "## \[Unreleased\]" "$CHANGELOG"; then
    sed -i "s/## \[Unreleased\]/## [Unreleased]\n\n## [${NEXT}] - ${TODAY}/" "$CHANGELOG"
    echo "CHANGELOG.md → [${NEXT}] section added"
else
    echo "WARNING: [Unreleased] header not found in CHANGELOG.md — skipping"
fi

# Update Unreleased comparison link
if grep -q "^\[Unreleased\]:.*" "$CHANGELOG"; then
    sed -i "s|^\[Unreleased\]:.*|[Unreleased]: https://github.com/olibartfast/neuriplo-kserve-runtime/compare/v${NEXT}...HEAD|" "$CHANGELOG"
else
    echo "[Unreleased]: https://github.com/olibartfast/neuriplo-kserve-runtime/compare/v${NEXT}...HEAD" >> "$CHANGELOG"
fi

# Add new version comparison link
if ! grep -q "^\[${NEXT}\]:" "$CHANGELOG"; then
    echo "[${NEXT}]: https://github.com/olibartfast/neuriplo-kserve-runtime/compare/v${CUR}...v${NEXT}" >> "$CHANGELOG"
fi

# Commit
git add VERSION CHANGELOG.md
git commit -m "Release v${NEXT}"
echo "Committed release v${NEXT}"

# Merge to master + tag
git checkout master
git merge --no-ff "$BRANCH" -m "Merge release/${NEXT} into master"
git tag -a "v${NEXT}" -m "Release v${NEXT}"
echo "Merged to master, tagged v${NEXT}"

# Merge back to develop
git checkout develop
git merge --no-ff "$BRANCH" -m "Merge release/${NEXT} back into develop"
echo "Merged back to develop"

# Clean up
git branch -d "$BRANCH"
echo "Deleted local branch $BRANCH"

# Push
git push origin master develop --tags
git push origin --delete "$BRANCH" 2>/dev/null || true
echo "Pushed master, develop, and tags to origin"

echo
echo "✔ Release v${NEXT} complete"

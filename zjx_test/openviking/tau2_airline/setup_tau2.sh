#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-${ROOT_DIR}/.external/tau2-bench}"
REF="${TAU2_REF:-refs/pull/297/head}"

if [[ ! -d "${TARGET}/.git" ]]; then
  git clone https://github.com/sierra-research/tau2-bench.git "${TARGET}"
fi

git -C "${TARGET}" fetch origin "${REF}"
git -C "${TARGET}" checkout --detach FETCH_HEAD
python -m pip install -e "${TARGET}"

cat > "${ROOT_DIR}/.env.tau2" <<EOF
export TAU2_REPO='${TARGET}'
EOF

echo "TAU-2 ready at ${TARGET}"
echo "Pinned commit: $(git -C "${TARGET}" rev-parse HEAD)"
echo "Run: source '${ROOT_DIR}/.env.tau2'"

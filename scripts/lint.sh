#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=${SCRIPT_DIR}/..

(
    cd ${REPO_DIR}
    FILES=$(git ls-files '*.py')
    pre-commit run black --files ${FILES[@]}
    pre-commit run ruff --files ${FILES[@]}
    pre-commit run mypy --files ${FILES[@]}
)

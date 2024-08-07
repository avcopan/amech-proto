#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=${SCRIPT_DIR}/..

(
    cd ${REPO_DIR}
    pytest -v --ignore=src/autochem
)

#!/bin/bash

# This script runs a command inside each repo in src/
#
# Arguments:
#   - Command to run (default: git status)

set -e  # if any command fails, quit
REPOS=("autochem")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=${SCRIPT_DIR}/..

# 0. Read arguments
COMMAND="${@:-git status}"

# 1. If running something other than git status, confirm
if [[ "${COMMAND}" != "git status" ]]; then
    read -p "Execute '${COMMAND}' in each repository? Press enter to confirm "
fi

# 2. Loop through each repo and execute the command
for repo in ${REPOS[@]}
do
    printf "\n*** Running command in ${REPO_DIR}/src/${repo} ***\n"
    (
        cd ${REPO_DIR}/src/${repo} && ${COMMAND}
    )
    printf "******\n"
done

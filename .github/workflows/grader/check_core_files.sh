#!/usr/bin/env bash
set -euo pipefail
source grader/check_common.sh

check_file "run_lab1.py"
check_file "README.md"

echo "OK: core files are present"

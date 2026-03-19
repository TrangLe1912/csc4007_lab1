#!/usr/bin/env bash
set -euo pipefail
source grader/check_common.sh

check_file "datacard/heuristics_scorecard.md"
check_file "datacard/metadata_register.md"

echo "OK: required datacard files are present"

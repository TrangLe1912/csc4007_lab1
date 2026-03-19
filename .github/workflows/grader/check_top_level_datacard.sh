#!/usr/bin/env bash
set -euo pipefail
source grader/check_common.sh

check_file "data_card.md"

echo "OK: top-level data card is present"

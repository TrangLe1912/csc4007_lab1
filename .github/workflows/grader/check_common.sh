#!/usr/bin/env bash
set -euo pipefail

check_file () {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "❌ Missing file: $path"
    exit 1
  fi
  echo "✅ Found file: $path"
}

check_dir () {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "❌ Missing directory: $path"
    exit 1
  fi
  echo "✅ Found directory: $path"
}

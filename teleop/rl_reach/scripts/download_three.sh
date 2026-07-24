#!/usr/bin/env bash
# Fetch the vendored three.js files for the reach visualizer.
#
# The visualizer (teleop/rl_reach/static/index.html) uses three.js r128 loaded
# as plain <script> tags (global THREE + the classic non-module OrbitControls),
# so no bundler / import-map is needed. These files are gitignored and fetched
# here once, like the MediaPipe model bundles.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p static/vendor
cd static/vendor

ver="0.128.0"
base="https://unpkg.com/three@${ver}"

fetch() {
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "  have  $out"
    return
  fi
  echo "  get   $out"
  curl -sSL --fail -o "$out" "$url"
}

echo "Downloading three.js r128 into static/vendor/ ..."
fetch "$base/build/three.min.js"                          three.min.js
fetch "$base/examples/js/controls/OrbitControls.js"       OrbitControls.js

echo
ls -lh ./*.js
echo
echo "Done."

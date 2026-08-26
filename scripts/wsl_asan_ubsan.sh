#!/bin/bash
# ASan+UBSan build and run of the C++ test suite + oracle diff + one edge_scorer
# invocation (TSan is skipped: this codebase has no threading, see README).
set -e

WROOT="/mnt/c/Users/Manas/Downloads/Total Job Application process flow/projects/early-warning-alarm-prediction"
BUILD_HOME="$HOME/build/early-warning-alarm-prediction-asan"

mkdir -p "$BUILD_HOME"
rsync -a --delete "$WROOT/cpp/" "$BUILD_HOME/cpp/"

cd "$BUILD_HOME/cpp"
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN_UBSAN=ON
cmake --build . -j4

echo "=== test_model (ASan+UBSan) ==="
./test_model "$WROOT/weights/weights.bin" "$WROOT/weights/norm.txt"

echo ""
echo "=== oracle_diff (ASan+UBSan) ==="
./oracle_diff "$WROOT/weights/weights.bin" "$WROOT/weights/norm.txt" "$WROOT/docs/oracle_windows.bin" "$WROOT/docs/oracle_targets.txt"

echo ""
echo "=== edge_scorer on one real eval run (ASan+UBSan) ==="
./edge_scorer "$WROOT/weights/weights.bin" "$WROOT/weights/norm.txt" \
  "$WROOT/data/eval_runs/upset_001_seed900001.bin" 180 0.90 3 \
  "/tmp/asan_smoke_pred.json" asan_smoke_run

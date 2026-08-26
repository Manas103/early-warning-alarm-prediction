#!/bin/bash
# Builds the C++ side natively in WSL (not on /mnt/c, for speed) and runs
# the unit test suite + oracle diff. Copies nothing back except what the
# caller explicitly captures from stdout; raw output should be redirected
# by the invoker into docs/.
set -e

WROOT="/mnt/c/Users/Manas/Downloads/Total Job Application process flow/projects/early-warning-alarm-prediction"
BUILD_HOME="$HOME/build/early-warning-alarm-prediction"

mkdir -p "$BUILD_HOME"
rsync -a --delete "$WROOT/cpp/" "$BUILD_HOME/cpp/"

cd "$BUILD_HOME/cpp"
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4

echo "=== test_model ==="
./test_model "$WROOT/weights/weights.bin" "$WROOT/weights/norm.txt"

echo ""
echo "=== oracle_diff ==="
./oracle_diff "$WROOT/weights/weights.bin" "$WROOT/weights/norm.txt" "$WROOT/docs/oracle_windows.bin" "$WROOT/docs/oracle_targets.txt"

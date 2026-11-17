#!/bin/sh
# Interleaves the C and Velt kernel benchmarks (the machine may be loaded, so each is run
# several times in turn and bench/best.py keeps the best per kernel).
# Usage: sh bench/run_kernels.sh <iterations> > results.txt
set -e
cd "$(dirname "$0")/.."
ITER=${1:-3}
for i in $(seq "$ITER"); do
  for size in "5000 384 15" "100000 384 7"; do
    set -- $size
    echo "## iter $i n=$1"
    uptime
    target/kernels_c "$1" "$2" "$3" 2>/dev/null
    target/kernels_velt "$1" "$2" "$3" | sed 's/^| Velt /| Velt(default) /'
    target/kernels_velt_native "$1" "$2" "$3" | sed 's/^| Velt /| Velt(native) /'
  done
done

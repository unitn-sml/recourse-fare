#!/bin/bash
set -euo pipefail

CONFIGS=(
"1"
"2"
"3"
"4"
"5"
"6"
"7"
"8"
"9"
"10"
)

for c in ${CONFIGS[@]}; do
  qsub -V -N "adult_${c}" -v COUNTER="${c}" launch_adult.sh
done
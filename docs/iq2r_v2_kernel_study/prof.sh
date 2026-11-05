#!/usr/bin/env bash
# usage (host): GPU=3 prof.sh NAME FMT INTER M "PMC ..." [REGEX]  -> prof/NAME/*counter_collection.csv (gate stage only)
N=$1 F=$2 I=$3 M=$4 P=$5 R=${6:-nobarrier}
GPU=${GPU:-3} ./optest.sh prof_$N "STAGES=gate REPS=1 CHECK=0 rocprofv3 --pmc $P --kernel-include-regex '$R' -d \$D/prof/$N -o run --output-format csv -- python3 \$D/vbench.py $F $I $M"

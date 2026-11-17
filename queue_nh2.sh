#!/bin/bash
cd "$(dirname "$0")/.."; mkdir -p logs; export OMP_NUM_THREADS=1
for pe in "133 1200" "40 500" "13.3 250"; do set -- $pe; out=results/p2_local_p${1}_nh2.pkl; [ -f $out ] || { echo "[$(date +%H:%M)] p2 NH=2 $1"; NH=2 nice -n 3 python3 scripts/p2_localscan.py $1 $2 > logs/p2_nh2_p$1.log 2>&1; :; }; done
echo "[$(date +%H:%M)] NH2 DONE"

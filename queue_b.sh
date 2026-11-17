#!/bin/bash
# Queue B (mixtures, 2-4 GB each, strictly one at a time; air only after queue A has finished)
cd "$(dirname "$0")/.."; mkdir -p logs; export OMP_NUM_THREADS=1 ION_MODEL=kin MIX_NE=200 MIX_UMAX=25 MIX_M=12 ITERS=18 ITERS2=16 MIX_TAG=_kin P7_FROM_OPT=1 RELAX=0.5 PHIRELAX=0.35
unset ION_K0
for g in "N2=2.25" "Ar:0.9,N2:0.1=1.25" "Ar:0.5,N2:0.5=1.88"; do echo "[$(date +%H:%M)] p7 $g"; python3 scripts/p7_queue.py "$g" >> logs/p7_queue_kin.log 2>&1; :; done
while [ ! -f results/queueA.done ]; do sleep 60; done
echo "[$(date +%H:%M)] p7 air"; python3 scripts/p7_queue.py "N2:0.79,O2:0.21=2.45" >> logs/p7_queue_kin.log 2>&1; :
echo "[$(date +%H:%M)] QUEUE B DONE"; touch results/queueB.done

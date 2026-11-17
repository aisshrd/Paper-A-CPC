#!/bin/bash
# Queue C: light post-processing for Paper 7: local model at each converged point, MC in the N2 fields, m=16 check for N2.
cd "$(dirname "$0")/.."; mkdir -p logs; export OMP_NUM_THREADS=1
nearest() { python3 - "$1" <<'PY'
import sys,glob,pickle,numpy as np
fs=[f for f in glob.glob(sys.argv[1]) if 'ckpt' not in f]
best=min(fs,key=lambda f: abs(pickle.load(open(f,'rb'))['balance']-1)) if fs else ''
print(best)
PY
}
# wait for the two N2 points
while [ $(ls results/discharge_mix_N2_p13.3_B*_kin.pkl 2>/dev/null | grep -v ckpt | wc -l) -lt 2 ]; do sleep 120; done
N2=$(nearest 'results/discharge_mix_N2_p13.3_B*_kin.pkl'); B=$(python3 -c "import pickle;print(pickle.load(open('$N2','rb'))['B_ext'])")
[ -f results/mc_n2_disc_B${B}_kin.pkl ] || { echo "[$(date +%H:%M)] MC N2 in $N2"; MCGAS=N2 MCREF=$N2 MCOUT=results/mc_n2_disc_B${B}_kin.pkl python3 scripts/mc_cyl_mag.py 3000 1 4e-5 > logs/mc_n2_kin.log 2>&1; :; }
# local model for every converged point of every gas (fast), repeated until queue B is done
while true; do
  for f in results/discharge_mix_*_kin.pkl results/discharge_te_B0.74_n1e+17_L1_m16_o2_kin_teff2.pkl; do
    case $f in *ckpt*) continue;; esac; [ -f "$f" ] || continue
    tagf=results/p7_local_$(basename $f .pkl).pkl
    [ -f $tagf ] || { python3 scripts/p7_local.py $f > logs/p7_local_$(basename $f .pkl).log 2>&1; g=$(grep -o "results/p7_local_[^ ]*\.pkl" logs/p7_local_$(basename $f .pkl).log | head -1); python3 - "$f" <<'PY'
import sys,pickle,os
f=sys.argv[1]; d=pickle.load(open(f,'rb')); mix=d.get('mix',[('Ar',1.0)]); gas='_'.join(f'{s}{x:g}' for s,x in mix); src=f'results/p7_local_{gas}.pkl'
if os.path.exists(src): os.replace(src, f'results/p7_local_{os.path.basename(f)}')
PY
    }
  done
  [ -f results/queueB.done ] && break; sleep 300
done
# radial-grid check for nitrogen: 16 cells at the same current as the nearest point
N2=$(nearest 'results/discharge_mix_N2_p13.3_B*_kin.pkl'); B=$(python3 -c "import pickle;print(pickle.load(open('$N2','rb'))['B_ext'])")
[ -f results/discharge_mix_N2_p13.3_B${B}_kin_m16.pkl ] || { echo "[$(date +%H:%M)] N2 m=16 check"; env -u ION_K0 ION_MODEL=kin MIX_NE=200 MIX_UMAX=25 MIX_M=16 TAG=_kin_m16 MIX_START=$N2 RELAX=0.5 PHIRELAX=0.35 python3 scripts/discharge_mix.py N2 13.3 $B 16 > logs/dmix_N2_m16.log 2>&1; :; }
echo "[$(date +%H:%M)] QUEUE C DONE"; touch results/queueC.done

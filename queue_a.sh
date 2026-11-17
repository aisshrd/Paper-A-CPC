#!/bin/bash
# Queue A (light jobs, ~0.6-2 GB each, serial): argon v2 scan, v1-quadrature demos, magnetized kin, p2 Nh=2, MC, p3 LMEA. Resumable: skips existing outputs.
cd "$(dirname "$0")/.."; mkdir -p logs; export OMP_NUM_THREADS=1 ION_MODEL=kin ION_K0=1.53
TE() { # out Bg n0 iters TAG [extra env...]
  out=$1; Bg=$2; n0=$3; it=$4; tag=$5; shift 5
  if [ -f "$out" ]; then echo "skip $out"; return; fi
  echo "[$(date +%H:%M)] run $out"; env "$@" TAG=$tag python3 scripts/discharge_te.py $Bg $n0 1 16 120 $it > logs/$(basename $out .pkl).log 2>&1; :; }
K74=results/discharge_te_B0.74_n1e+17_L1_m16_o2_kin_teff2.pkl; OLD7=results/discharge_te_B0.7_n1e+17_L1_m16_o2.pkl
while pgrep -f "discharge_te.py 0.74" >/dev/null; do sleep 30; done
TE $K74 0.74 1e17 30 _kin_teff2 RELAX=0.5 PHIRELAX=0.35 START=$OLD7
for B in 0.7 0.76 0.8; do TE results/discharge_te_B${B}_n1e+17_L1_m16_o2_kin_teff2.pkl $B 1e17 30 _kin_teff2 RELAX=0.5 PHIRELAX=0.35 START=$K74; done
# v1 (midpoint) quadrature demonstrations for Paper 4
TE results/discharge_te_B0.74_n1e+17_L1_m16_o2_kin_v1warm.pkl 0.74 1e17 30 _kin_v1warm TEFF_V2=0 RELAX=0.5 PHIRELAX=0.35 START=$OLD7
TE results/discharge_te_B0.74_n1e+17_L1_m16_o2_kin_v1_r0605.pkl 0.74 1e17 25 _kin_v1_r0605 TEFF_V2=0 RELAX=0.6 PHIRELAX=0.5 START=$OLD7
TE results/discharge_te_B0.74_n1e+17_L1_m16_o2_kin_v1_pr02.pkl 0.74 1e17 20 _kin_v1_pr02 TEFF_V2=0 RELAX=0.5 PHIRELAX=0.2 START=$OLD7
TE results/discharge_te_B0.74_n1e+17_L1_m16_o2_kin_v1cold.pkl 0.74 1e17 40 _kin_v1cold TEFF_V2=0 RELAX=0.5 PHIRELAX=0.35 COLD=1
TE results/discharge_te_B0.74_n1e+17_L1_m16_o2_and4_kin_v1.pkl 0.74 1e17 40 _kin_v1 TEFF_V2=0 ANDERSON=4 RELAX=0.5 PHIRELAX=0.35 START=$OLD7
TE results/discharge_te_B0.74_n3e+17_L1_m16_o2_kin_v1.pkl 0.74 3e17 40 _kin_v1 TEFF_V2=0 RELAX=0.5 PHIRELAX=0.35 COLD=1
# v2 (exact-integral) counterparts
TE results/discharge_te_B0.74_n1e+17_L1_m16_o2_kin_teff2_cold.pkl 0.74 1e17 40 _kin_teff2_cold RELAX=0.5 PHIRELAX=0.35 COLD=1
TE results/discharge_te_B0.74_n1e+17_L1_m16_o2_and4_kin_teff2.pkl 0.74 1e17 20 _kin_teff2 ANDERSON=4 RELAX=0.5 PHIRELAX=0.35 START=$K74
TE results/discharge_te_B0.74_n3e+17_L1_m16_o2_kin_teff2.pkl 0.74 3e17 30 _kin_teff2 RELAX=0.5 PHIRELAX=0.35 COLD=1
# magnetized, corrected closure, at the argon operating point 0.74 G
MAG() { out=$1; B0=$2; Bg=$3; it=$4; rx=$5; px=$6; start=$7; erstart=$8
  if [ -f "$out" ]; then echo "skip $out"; return; fi
  echo "[$(date +%H:%M)] run $out"; MAG_TAG=_kin MAG_START=$start MAG_ER_START=$erstart RELAX=$rx PHIRELAX=$px python3 scripts/discharge_mag.py $B0 $Bg $it > logs/$(basename $out .pkl).log 2>&1; :; }
M0=results/discharge_mag_B00_Bext0.74_kin.pkl; M10=results/discharge_mag_B010_Bext0.74_kin.pkl; M20=results/discharge_mag_B020_Bext0.74_kin.pkl; M40=results/discharge_mag_B040_Bext0.74_kin.pkl
MAG $M0 0 0.74 20 0.5 0.35 $K74 ""
MAG $M10 10 0.74 25 0.5 0.35 $K74 $M0
MAG $M20 20 0.74 30 0.4 0.25 $M10 $M10
MAG $M40 40 0.74 40 0.25 0.15 $M20 $M20
MAG results/discharge_mag_B010_Bext0.68_kin.pkl 10 0.68 25 0.5 0.35 $M10 $M10
MAG results/discharge_mag_B020_Bext0.66_kin.pkl 20 0.66 30 0.4 0.25 $M20 $M20
MAG results/discharge_mag_B040_Bext0.6_kin.pkl 40 0.60 40 0.25 0.15 $M40 $M40
# Paper 2: second time harmonic in the prescribed skin-layer field
for pe in "133 1200" "40 500" "13.3 250"; do set -- $pe; out=results/p2_local_p${1}_nh2.pkl; [ -f $out ] || { echo "[$(date +%H:%M)] p2 NH=2 $1"; NH=2 python3 scripts/p2_localscan.py $1 $2 > logs/p2_nh2_p$1.log 2>&1; :; }; done
# MC with refined time step (Papers 1,3) and the N2 discharge MC (Paper 7)
while pgrep -f 'mc_cyl_real.py 3000 41' >/dev/null; do sleep 60; done; [ -f results/mc_cyl_real_N3000_dt4_80us_s41.pkl ] || { echo "[$(date +%H:%M)] MC dt/4"; MCDTFAC=4 MCOUT=results/mc_cyl_real_N3000_dt4_80us_s41.pkl python3 scripts/mc_cyl_real.py 3000 41 8e-5 > logs/mc_dt4.log 2>&1; :; }
[ -f results/p3_cylcoef_lmea.flag ] || { python3 scripts/p3_lmea_sigma.py > logs/p3_lmea.log 2>&1 && touch results/p3_cylcoef_lmea.flag; :; }
echo "[$(date +%H:%M)] QUEUE A DONE"; touch results/queueA.done

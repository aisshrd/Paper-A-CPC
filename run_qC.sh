while pgrep -f "run_qA.sh" > /dev/null || pgrep -f "run_qB.sh" > /dev/null; do sleep 20; done
python3 scripts/slab_sc.py 13.3 1e16 8 3 80 48 40 _prod

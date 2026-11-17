while kill -0 333 2>/dev/null; do sleep 15; done
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 scripts/slab_sc.py 40 1e16 8 2 80 48 40 _prod
python3 scripts/slab_sc.py 133 1e16 8 2 80 48 40 _prod
python3 scripts/slab_sc.py 13.3 1e16 12 3 80 48 40 _prod

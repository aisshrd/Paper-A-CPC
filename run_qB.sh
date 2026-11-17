export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 scripts/slab_sc.py 13.3 1e16 4 2 80 48 40 _prod
python3 scripts/slab_sc.py 13.3 1e16 8 3 80 48 40 _prod
python3 scripts/slab_sc.py 13.3 3e16 8 2 80 48 40 _prod

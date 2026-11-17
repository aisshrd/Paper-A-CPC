while kill -0 9477 2>/dev/null; do sleep 15; done
python3 scripts/slab_sc.py 13.3 1e16 8 3 80 48 40 _prod
python3 scripts/slab_sc.py 13.3 1e16 12 3 80 48 40 _prod
python3 scripts/slab_sc.py 13.3 3e16 8 2 80 48 40 _prod

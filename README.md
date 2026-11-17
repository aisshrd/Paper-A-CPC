# Data

| file | content |
|---|---|
| `Ar_LXCat.txt`, `N2_LXCat.txt`, `O2_LXCat.txt` | **not included — download them (see below)** |
| `reference_profiles_p13_n1e17_B1.3G.json` | reference fields of the converged self-consistent point (13.3 Pa, n0 = 1e17 m^-3, B_ext = 1.3 G, solver generation II): r/R, E_theta amplitude and phase, B_z amplitude and phase, phi, T_e, n_e, n_i, nu_iz, balance. These fields are imposed in the cylindrical Monte Carlo and in the solver comparisons (article §4.3). |

## Cross sections

The code uses the IST-Lisbon cross-section sets from LXCat: argon (elastic momentum transfer, 37 excitation levels,
ionization), nitrogen (effective momentum transfer, 10 vibrational levels, 13 electronic states, 2 ionization channels) and
oxygen (effective momentum transfer, 4 vibrational levels, a1Δg, b1Σg+, Herzberg states, 2 dissociation channels, 2 higher
electronic losses, ionization, dissociative attachment). The files are distributed by LXCat under its own terms and are
therefore not included here.

Obtain them in either of two ways and save them as `data/Ar_LXCat.txt`, `data/N2_LXCat.txt`, `data/O2_LXCat.txt`
(or set the environment variable `LXCAT_DIR` to the `Code/Input/` folder of LoKI-B):

1. www.lxcat.net → Download → database **IST-Lisbon**, species **Ar**, **N2**, **O2**, all processes, format "LXCat".
2. From the LoKI-B repository (github.com/IST-Lisbon/LoKI): `Code/Input/Argon/Ar_LXCat.txt`, `Code/Input/Nitrogen/N2_LXCat.txt`,
   `Code/Input/Oxygen/O2_LXCat.txt`.

Please cite the database as requested in the file header
("IST-Lisbon database, www.lxcat.net, retrieved on ...").

"""Gas mixtures for the cylindrical total-energy solver: build a composite cross-section dict (the 'xs' of CylKineticsTE) from
the species files of ekin_mix.load_species, with mole fractions folded into the cross sections.
  elastic : sum_s x_s sigma_m,s (momentum transfer; EFFECTIVE if no ELASTIC)
  eloss   : sum_s x_s (m/M_s) sigma_s   (elastic energy loss)
  inel    : [(threshold, x_s*sigma, 'species:product'), ...]
  iz      : [(threshold, x_s*sigma), ...]
  att     : [x_s*sigma, ...]
Ion parameters for the ambipolar closure: mass of the dominant ion and a crude ion-atom cross section."""
import numpy as np
from ekin_mix import load_species, FILES, CAR, A0
UG=np.concatenate([[0.0],np.logspace(-3,np.log10(300.),1200)])
def _on(d,u): return np.interp(u,d[:,0],d[:,1],left=d[0,1],right=d[-1,1])
def mixture_xs(mix, car=True):
    """mix: list of (species name or dict, mole fraction); car: continuous rotational loss (as in ekin_mix).
    Species with only an EFFECTIVE cross section: elastic = effective - sum of inelastic (as in ekin_mix)."""
    el=np.zeros_like(UG); elo=np.zeros_like(UG); inel=[]; iz=[]; att=[]
    for sp,x in mix:
        s=load_species(sp) if isinstance(sp,str) else sp
        if s['el'] is not None: sm=_on(s['el'],UG)
        else:
            sin=sum((_on(d,UG)*(UG>th) for th,d,_ in s['inel']),np.zeros_like(UG))+sum((_on(d,UG)*(UG>th) for th,d,_ in s['iz']),np.zeros_like(UG))+sum((_on(d,UG) for d,_ in s['att']),np.zeros_like(UG))
            sm=np.maximum(_on(s['eff'],UG)-sin,0.0)
        el+=x*sm; elo+=x*s['mM']*sm
        if car and s['name'] in CAR:                       # continuous rotational loss: W = -4 B s0 x N gamma u  ->  (m/M)sigma_eff = 2 B s0 / u
            B,q=CAR[s['name']]; s0=8*np.pi*q**2*A0**2/15; elo+=x*2*B*s0/np.maximum(UG,1e-4)
        for th,d,prod in s['inel']: inel.append((th,np.c_[d[:,0],x*d[:,1]],f"{s['name']}:{prod}"))
        for th,d,prod in s['iz']: iz.append((th,np.c_[d[:,0],x*d[:,1]]))
        for d,prod in s['att']: att.append(np.c_[d[:,0],x*d[:,1]])
    return dict(elastic=np.c_[UG,el],eloss=np.c_[UG,elo],inel=inel,iz=iz,att=att,mM=None,mix=[(sp if isinstance(sp,str) else sp['name'],x) for sp,x in mix])
ION={'Ar':(39.95,1e-18),'N2':(28.01,1e-18),'O2':(32.0,1e-18),'He':(4.0,0.5e-18)}
def ion_params(mix):
    """dominant ion = the species with the largest mole fraction (crude): (M_amu, sigma_i)"""
    name=max(mix,key=lambda t:t[1])[0]; name=name if isinstance(name,str) else name['name']; return ION[name]

# ----- ion mobility for the corrected closure (v1.7.0): measured K0 for resonant pairs (Ellis et al. 1976), Langevin for the others, Blanc's law
K0_RES={('Ar','Ar'):1.53,('N2','N2'):1.87,('O2','O2'):2.24,('He','He'):10.4}     # cm^2/(V s) at 2.6868e25 m^-3, low field
POL={'Ar':1.64,'N2':1.74,'O2':1.58,'He':0.205}                                    # polarizability, A^3
MASS={'Ar':39.95,'N2':28.01,'O2':32.00,'He':4.00}
EIZ={'Ar':15.76,'N2':15.58,'O2':12.07,'He':24.59}
def ion_mobility(mix):
    """Dominant ion = species with the lowest ionization threshold among those with mole fraction >= 0.1 (charge transfer drives
    the ion population to the lowest threshold). K0(ion in gas j): measured for the resonant pair, Langevin K0=13.853/sqrt(alpha_j mu_ij)
    otherwise; the mixture value from Blanc's law 1/K0 = sum_j x_j/K0_j.  Returns (ion name, M_amu, K0_mix [cm^2/Vs])."""
    names=[(sp if isinstance(sp,str) else sp['name'],x) for sp,x in mix]
    cand=[n for n,x in names if x>=0.1] or [max(names,key=lambda t:t[1])[0]]
    ion=min(cand,key=lambda n:EIZ[n]); M=MASS[ion]; inv=0.0
    for g,x in names:
        if (ion,g) in K0_RES: K=K0_RES[(ion,g)]
        else: mu=M*MASS[g]/(M+MASS[g]); K=13.853/np.sqrt(POL[g]*mu)
        inv+=x/K
    return ion,M,1.0/inv

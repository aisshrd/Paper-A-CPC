import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, pickle, sys
m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb'))
uc,h=m1['eedf']; h=(h+m2['eedf'][1])/2; du=0.5
d=pickle.load(open(sys.argv[1],'rb')); X=d['X']; ec=d['ec']; F0=d['F0']; uk=d['ukin']
def te_eepf(ksel):
    # average EEPF over cells ksel on a kinetic-energy grid: f(u) = F0 at u -> interpolate per cell, weight by disk area
    ug=uc; out=np.zeros_like(ug)
    for k in ksel:
        ok=uk[:,k]>0; out+=np.interp(ug,uk[ok,k],F0[ok,k],left=0,right=0)*(1 if True else 1)
    out/=len(ksel); n=np.sum(out*np.sqrt(ug)*du); return out/n
for b,lab,sel in [(0,'wall 0.9-1',X>0.9),(1,'mid 0.4-0.6',(X>0.4)&(X<0.6)),(2,'axis 0-0.2',X<0.2)]:
    fm=h[b]/np.sum(h[b])/du/np.sqrt(uc); ft=te_eepf(np.where(sel)[0])
    print(lab,'   u    MC EEPF   TE EEPF  ratio')
    for i in [0,2,4,8,12,16,20,24,28,32,36]:
        print(f'   {uc[i]:5.2f}  {fm[i]:.3e}  {ft[i]:.3e}  {fm[i]/max(ft[i],1e-30):.2f}')
    # local Te from EEPF slopes 1-8 eV
    def tfit(f,lo,hi): s=(uc>lo)&(uc<hi)&(f>0); p=np.polyfit(uc[s],np.log(f[s]),1); return -1/p[0]
    print('   slope temperature 1-8 eV: MC %.2f TE %.2f ; 12-20 eV: MC %.2f TE %.2f'%(tfit(fm,1,8),tfit(ft,1,8),tfit(fm,12,20),tfit(ft,12,20)))

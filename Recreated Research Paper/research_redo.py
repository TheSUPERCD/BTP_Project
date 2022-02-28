from numpy.core.numeric import Inf
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

from tmm import (coh_tmm, unpolarized_RT, ellips,
                       position_resolved, find_in_structure_with_inf)

eV = 1.602e-19
q = 1.602e-19
h = 6.62e-34
c = 3e8
k_B = 1.38e-23
T = 300

Title = 'Perovskite (CH3NH3PbBr3 - microcrystalline [Bri16])'
K_lst = list(pd.read_csv('Perovskite (CH3NH3PbBr3 - microcrystalline [Bri16]).csv')['k'])
n_lst = list(pd.read_csv('Perovskite (CH3NH3PbBr3 - microcrystalline [Bri16]).csv')['n'])
alpha_lst = list(pd.read_csv('Perovskite (CH3NH3PbBr3 - microcrystalline [Bri16]).csv')['α (cm⁻¹)'])
start_lmda = 400
gap_lmda = 1
E_g = 2.2*eV

# Title = 'Perovskite (CH3NH3PbI3 (MAPI), Eg=1.557 eV [Man18])'
# K_lst = list(pd.read_csv('Perovskite (CH3NH3PbI3 (MAPI), Eg=1.557 eV [Man18]).csv')['k'])
# n_lst = list(pd.read_csv('Perovskite (CH3NH3PbI3 (MAPI), Eg=1.557 eV [Man18]).csv')['n'])
# alpha_lst = list(pd.read_csv('Perovskite (CH3NH3PbI3 (MAPI), Eg=1.557 eV [Man18]).csv')['α (cm⁻¹)'])
# print(len(alpha_lst))
# start_lmda = 300
# gap_lmda = 5
# E_g = 1.55*eV


lmda_max = h*c/E_g
print('Max wavelegth is =', lmda_max*1e9, 'nm')
W=300e-9
V = 1.5

def K(lmda):
    return K_lst[int((lmda*1e9 - start_lmda)/gap_lmda)]

def n(lmda):
    return n_lst[int((lmda*1e9 - start_lmda)/gap_lmda)]

def alpha(lmda):
    return alpha_lst[int((lmda*1e9 - start_lmda)/gap_lmda)]*100

def a(lmda, thickness=W,case='null'):
    if case == 'beer-lambert':
        return 1 - np.exp(-2*alpha(lmda)*thickness)
    
    elif case == 'TMM':
        d_list = [np.inf,W*1e9,np.inf] #in nm
        n_list = [1, complex(n(lmda), K(lmda)), 1]
        c_tmm = coh_tmm('s',n_list,d_list,0,lmda*1e9) #in nm
        return 1 - c_tmm['T'] - c_tmm['R']
    
    else:
        if lmda < lmda_max:
            return 1
        else:
            return 0

def Responsivity(lmda, case='null'):
    return q*lmda*EQE(lmda, case)/(h*c)

def IQE(lmda):
    return 1

def EQE(lmda, case):
    return a(lmda, thickness=W, case=case) * IQE(lmda)




lmda = list(range(start_lmda, int(lmda_max*1e9 + 10), gap_lmda))





absorb = [0]*len(lmda)
for i in range(0, len(lmda)):
    absorb[i] = a(lmda[i]*1e-9)
plt.plot(lmda, absorb, label='Ideal')
plt.xlabel('Wavelength(nm)-->')
plt.ylabel('Absorptivity-->')

absorb_beer_lambert = [0]*len(lmda)
for i in range(0, len(lmda)):
    absorb_beer_lambert[i] = a(lmda[i]*1e-9, case='beer-lambert')
plt.plot(lmda, absorb_beer_lambert, label='Beer-Lambert')

absorb_tmm = [0]*len(lmda)
for i in range(0, len(lmda)):
    absorb_tmm[i] = a(lmda[i]*1e-9, case='TMM')
plt.plot(lmda, absorb_tmm, label='TMM')

plt.legend()
plt.title(Title)
plt.show()



resp = [0]*len(lmda)
for i in range(0, len(lmda)):
    resp[i] = Responsivity(lmda[i]*1e-9)
plt.plot(lmda, resp, label='Ideal')
plt.xlabel('Wavelength(nm)-->')
plt.ylabel('Responsivity-->')

resp_beer_lambert = [0]*len(lmda)
for i in range(0, len(lmda)):
    resp_beer_lambert[i] = Responsivity(lmda[i]*1e-9, case='beer-lambert')
plt.plot(lmda, resp_beer_lambert, label='Beer-Lambert')

resp_tmm = [0]*len(lmda)
for i in range(0, len(lmda)):
    resp_tmm[i] = Responsivity(lmda[i]*1e-9, case='TMM')
plt.plot(lmda, resp_tmm, label='TMM')

plt.legend()
plt.title(Title)
plt.show()
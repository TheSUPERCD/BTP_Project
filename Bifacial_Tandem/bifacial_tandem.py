from numpy.core.numeric import Inf
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

from tmm import (coh_tmm, unpolarized_RT, ellips,
                       position_resolved, find_in_structure_with_inf, absorp_in_each_layer)

eV = 1.602e-19
q = 1.602e-19
h = 6.62e-34
c = 3e8
k_B = 1.38e-23
T = 300

# Perovskite - main-cell 


K_lst_glass = list(pd.read_csv('Glass (Borosilicate [Sch]).csv')['k'])[10::2]
n_lst_glass = list(pd.read_csv('Glass (Borosilicate [Sch]).csv')['n'])[10::2]
alpha_lst_glass = list(pd.read_csv('Glass (Borosilicate [Sch]).csv')['α (cm⁻¹)'])[10::2]
print(alpha_lst_glass[0])

K_lst_tco = list(pd.read_csv('ITO (Sputtered 0.17e20 [Hol13]).csv')['k'])[20::2]
n_lst_tco = list(pd.read_csv('ITO (Sputtered 0.17e20 [Hol13]).csv')['n'])[20::2]
alpha_lst_tco = list(pd.read_csv('ITO (Sputtered 0.17e20 [Hol13]).csv')['α (cm⁻¹)'])[20::2]
print(alpha_lst_tco[0])


K_lst_etl = list(pd.read_csv('TiO2 (APCVD 150oC [Ric03]).csv')['k'])
n_lst_etl = list(pd.read_csv('TiO2 (APCVD 150oC [Ric03]).csv')['n'])
alpha_lst_etl = list(pd.read_csv('TiO2 (APCVD 150oC [Ric03]).csv')['α (cm⁻¹)'])
print(alpha_lst_etl[0])


K_lst_perov = list(pd.read_csv('Perovskite (CH3NH3PbI3 (MAPI), Eg=1.557 eV [Man18]).csv')['k'])[10::2]
n_lst_perov = list(pd.read_csv('Perovskite (CH3NH3PbI3 (MAPI), Eg=1.557 eV [Man18]).csv')['n'])[10::2]
alpha_lst_perov = list(pd.read_csv('Perovskite (CH3NH3PbI3 (MAPI), Eg=1.557 eV [Man18]).csv')['α (cm⁻¹)'])[10::2]
print(alpha_lst_perov[0])
E_g = 1.557*eV


K_lst_htl = list(pd.read_csv('Spiro-OMeTAD [Fil15].csv')['k'])
n_lst_htl = list(pd.read_csv('Spiro-OMeTAD [Fil15].csv')['n'])
alpha_lst_htl = list(pd.read_csv('Spiro-OMeTAD [Fil15].csv')['α (cm⁻¹)'])
print(alpha_lst_htl[0])


# Si - sub-cell 


K_lst_a_Si_n = [list(pd.read_csv('Si (Amorphous n [Hol12]).csv')['k'])[i] for i in [40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 60, 61, 62, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 74, 75]]
n_lst_a_Si_n = [list(pd.read_csv('Si (Amorphous n [Hol12]).csv')['n'])[i] for i in [40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 60, 61, 62, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 74, 75]]
alpha_lst_a_Si_n = [list(pd.read_csv('Si (Amorphous n [Hol12]).csv')['α (cm⁻¹)'])[i] for i in [40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 60, 61, 62, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 74, 75]]
print(alpha_lst_a_Si_n[0])

K_lst_a_Si_p = [list(pd.read_csv('Si (Amorphous p [Hol12]).csv')['k'])[i] for i in [40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 60, 61, 62, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 74, 75]]
n_lst_a_Si_p = [list(pd.read_csv('Si (Amorphous p [Hol12]).csv')['n'])[i] for i in [40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 60, 61, 62, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 74, 75]]
alpha_lst_a_Si_p = [list(pd.read_csv('Si (Amorphous p [Hol12]).csv')['α (cm⁻¹)'])[i]  for i in [40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 60, 61, 62, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 74, 75]]
print(alpha_lst_a_Si_p[0])


K_lst_a_Si_i = [list(pd.read_csv('Si (Amorphous i [Hol12]).csv')['k'])[i+1] for i in [40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 60, 61, 62, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 74, 75]] 
n_lst_a_Si_i = [list(pd.read_csv('Si (Amorphous i [Hol12]).csv')['n'])[i+1] for i in [40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 60, 61, 62, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 74, 75]]
alpha_lst_a_Si_i = [list(pd.read_csv('Si (Amorphous i [Hol12]).csv')['α (cm⁻¹)'])[i+1] for i in [40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 60, 61, 62, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 74, 75]]
print(alpha_lst_a_Si_i[0])


K_lst_c_Si = list(pd.read_csv('Si (Crystalline, 300 K [Gre08])(1) - c-Si(n).csv')['k'])[10:]
n_lst_c_Si = list(pd.read_csv('Si (Crystalline, 300 K [Gre08])(1) - c-Si(n).csv')['n'])[10:]
alpha_lst_c_Si = list(pd.read_csv('Si (Crystalline, 300 K [Gre08])(1) - c-Si(n).csv')['α (cm⁻¹)'])[10:]
print(alpha_lst_c_Si[0])
E_g = 1.557*eV



start_lmda = 350
gap_lmda = 10
lmda_max = h*c/E_g
print('Max wavelegth is =', lmda_max*1e9, 'nm')
W=300e-9
V = 1.5

def K(lmda, mat='perov'):
    if mat == 'glass':
        return K_lst_glass[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'tco':
        return K_lst_tco[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'etl':
        return K_lst_etl[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'htl':
        return K_lst_htl[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'a-Si(n)':
        return K_lst_a_Si_n[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'a-Si(p)':
        return K_lst_a_Si_p[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'a-Si(i)':
        return K_lst_a_Si_i[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'c-Si':
        return K_lst_c_Si[int((lmda*1e9 - start_lmda)/gap_lmda)]
    else:
        return K_lst_perov[int((lmda*1e9 - start_lmda)/gap_lmda)]

def n(lmda, mat='perov'):
    if mat == 'glass':
        return 1.5
    elif mat == 'tco':
        return n_lst_tco[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'etl':
        return n_lst_etl[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'htl':
        return n_lst_htl[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'a-Si(n)':
        return n_lst_a_Si_n[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'a-Si(p)':
        return n_lst_a_Si_p[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'a-Si(i)':
        return n_lst_a_Si_i[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'c-Si':
        return n_lst_c_Si[int((lmda*1e9 - start_lmda)/gap_lmda)]
    else:
        return n_lst_perov[int((lmda*1e9 - start_lmda)/gap_lmda)]

def alpha(lmda, mat='perov'):
    return alpha_lst_perov[int((lmda*1e9 - start_lmda)/gap_lmda)]*100

def a(lmda, thickness=W,case='null'):
    if case == 'beer-lambert':
        return 1 - np.exp(-2*alpha(lmda)*thickness)
    
    elif case == 'TMM':
        d_list = [np.inf, 
        100, # glass
        200, # ITO
        30, # TiO2 - etl
        W*1e9, # Perovskite
        300, # Spiro-MeTAD - htl
        200, # ITO
        20, # a-Si(n)
        5, # a-Si(i) 
        200, # c_Si
        5, # a-Si(i) 
        20, # a-Si(p)
        200, # ITO
        100, # glass
        np.inf] #in nm
        
        n_list = [1, 
        complex(1.5, K(lmda, mat='glass')), 
        complex(n(lmda, mat='tco'), K(lmda, mat='tco')), 
        complex(n(lmda, mat='etl'), K(lmda, mat='htl')), 
        complex(n(lmda), K(lmda)), 
        complex(n(lmda, mat='htl'), K(lmda, mat='etl')), 
        complex(n(lmda, mat='tco'), K(lmda, mat='tco')), 
        complex(n(lmda, mat='a-Si(n)'), K(lmda, mat='a-Si(n)')), 
        complex(n(lmda, mat='a-Si(i)'), K(lmda, mat='a-Si(i)')), 
        complex(n(lmda, mat='c-Si'), K(lmda, mat='c-Si')), 
        complex(n(lmda, mat='a-Si(i)'), K(lmda, mat='a-Si(i)')), 
        complex(n(lmda, mat='a-Si(p)'), K(lmda, mat='a-Si(p)')), 
        complex(n(lmda, mat='tco'), K(lmda, mat='tco')), 
        complex(1.5, K(lmda, mat='glass')), 
        1]
        c_tmm = coh_tmm('s',n_list,d_list,0,lmda*1e9) #in nm
        # return (1 - c_tmm['T'] - c_tmm['R']) # full-device absoptivity
        absp_layer = absorp_in_each_layer(c_tmm)
        # return (1 - c_tmm['T'] - c_tmm['R']) * (absp_layer[4] / (1- absp_layer[0] -absp_layer[14])) # perovskite layer absorption
        return (1 - c_tmm['T'] - c_tmm['R']) * (absp_layer[9] / (1- absp_layer[0] -absp_layer[14])) # c-Si layer absorption
    
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



# Title = 'Bi-facial Perovskite-Si Tandem Solar Cells (HIT) : Device Sim'
# Title = 'Bi-facial Perovskite-Si Tandem Solar Cells (HIT) : Layer Sim (Perovskite)'
Title = 'Bi-facial Perovskite-Si Tandem Solar Cells (HIT) : Layer Sim (c-Si)'
# Title = 'Bi-facial Perovskite-Si Tandem Solar Cells (HIT) : Reversed Device Sim'


lmda = list(range(start_lmda, 810, gap_lmda))



absorb = [0]*len(lmda)
for i in range(0, len(lmda)):
    absorb[i] = a(lmda[i]*1e-9)
plt.plot(lmda, absorb, label='Ideal')
plt.xlabel('Wavelength(nm)-->')
plt.ylabel('Absorptivity-->')

# absorb_beer_lambert = [0]*len(lmda)
# for i in range(0, len(lmda)):
#     absorb_beer_lambert[i] = a(lmda[i]*1e-9, case='beer-lambert')
# plt.plot(lmda, absorb_beer_lambert, label='Beer-Lambert')

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

# resp_beer_lambert = [0]*len(lmda)
# for i in range(0, len(lmda)):
#     resp_beer_lambert[i] = Responsivity(lmda[i]*1e-9, case='beer-lambert')
# plt.plot(lmda, resp_beer_lambert, label='Beer-Lambert')

resp_tmm = [0]*len(lmda)
for i in range(0, len(lmda)):
    resp_tmm[i] = Responsivity(lmda[i]*1e-9, case='TMM')
plt.plot(lmda, resp_tmm, label='TMM')

plt.legend()
plt.title(Title)
plt.show()
from cmath import exp
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


solar_spectra_atm_lst = list(pd.read_csv('AM0AM1_5.csv')['Global tilt  W*m-2*nm-1'])[70::10]
solar_spectra_ext_lst = list(pd.read_csv('AM0AM1_5.csv')['Extraterrestrial W*m-2*nm-1'])[70::10]
print(solar_spectra_atm_lst[0])
print(solar_spectra_ext_lst[0])

solar_atm = list(pd.read_csv('AM0AM1_5.csv')['Global tilt  W*m-2*nm-1'])
solar_ext = list(pd.read_csv('AM0AM1_5.csv')['Extraterrestrial W*m-2*nm-1'])
lmda_solar = list(pd.read_csv('AM0AM1_5.csv')['Wavelength (nm)'])[70:570:10]
solar_incident_atm = 0
solar_incident_ext = 0
for i in range(0, len(solar_atm)):
    solar_incident_atm += solar_atm[i]*1
    solar_incident_ext += solar_ext[i]*1



K_lst_glass = list(pd.read_csv('Glass (Borosilicate [Sch]).csv')['k'])[10::2]
n_lst_glass = list(pd.read_csv('Glass (Borosilicate [Sch]).csv')['n'])[10::2]
alpha_lst_glass = list(pd.read_csv('Glass (Borosilicate [Sch]).csv')['α (cm⁻¹)'])[10::2]
print(alpha_lst_glass[0])

K_lst_Ag = list(pd.read_csv('Ag (Pure [Jia16]).csv')['k'])[50::10]
n_lst_Ag = list(pd.read_csv('Ag (Pure [Jia16]).csv')['n'])[50::10]
alpha_lst_Ag = list(pd.read_csv('Ag (Pure [Jia16]).csv')['α (cm⁻¹)'])[50::10]
print(alpha_lst_Ag[0])

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



start_lmda = 350
gap_lmda = 10
lmda_max = h*c/E_g
print('Max wavelegth is =', lmda_max*1e9, 'nm')
W=300e-9
V = 1.01

def K(lmda, mat='perov'):
    if mat == 'glass':
        return K_lst_glass[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'tco':
        return K_lst_tco[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'etl':
        return K_lst_etl[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'htl':
        return K_lst_htl[int((lmda*1e9 - start_lmda)/gap_lmda)]
    elif mat == 'Ag':
        return K_lst_Ag[int((lmda*1e9 - start_lmda)/gap_lmda)]
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
    elif mat == 'Ag':
        return n_lst_Ag[int((lmda*1e9 - start_lmda)/gap_lmda)]
    else:
        return n_lst_perov[int((lmda*1e9 - start_lmda)/gap_lmda)]

def alpha(lmda, mat='perov'):
    return alpha_lst_perov[int((lmda*1e9 - start_lmda)/gap_lmda)]*100

def a(lmda, thickness=W,case='null'):
    if case == 'beer-lambert':
        return 1 - np.exp(-2*alpha(lmda)*thickness)
    
    elif case == 'TMM':
        d_list = [np.inf, 
        100, 
        200, 
        30, 
        thickness*1e9, 
        300, 
        200, 
        100, 
        np.inf] #in nm
        
        n_list = [1, 
        complex(1.5, K(lmda, mat='glass')), 
        complex(n(lmda, mat='tco'), K(lmda, mat='tco')), 
        complex(n(lmda, mat='etl'), K(lmda, mat='etl')), 
        complex(n(lmda), K(lmda)), 
        complex(n(lmda, mat='htl'), K(lmda, mat='htl')), 
        complex(n(lmda, mat='tco'), K(lmda, mat='tco')), 
        complex(1.5, K(lmda, mat='glass')), 
        1]
        c_tmm = coh_tmm('s',n_list,d_list,0,lmda*1e9) #in nm
        # return (1 - c_tmm['T'] - c_tmm['R']) # full-device absoptivity
        absp_layer = absorp_in_each_layer(c_tmm)
        return (1 - c_tmm['T'] - c_tmm['R']) * (absp_layer[4] / (1- absp_layer[0] -absp_layer[8])) # perovskite layer absorption
    
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



# Title = 'Bi-facial Perovskite Solar Cells (HIT) : Reversed Device Sim (Glass on)'
Title = 'Perovskite Solar Cells (HIT) : Layer Sim (Perovskite - Glass on)'


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



def loss_therm(d=W, solar_spectra=solar_atm[70::10]):
    lmda_f = list(range(350, 810, 10))
    loss_power = 0
    for i in range(0, len(lmda_f)):
        converted_power_step = a(lmda_f[i]*1e-9, thickness=d, case='TMM')*solar_spectra[i]*gap_lmda
        loss_power += converted_power_step*(1 - 1.5758*lmda_f[i]*1e-9/((h*c)/q)) # [1 - (Eg - Eu +kT).lmda / hc]
    return loss_power
print('Thermal Loss =', loss_therm())

def loss_rad_recomb(d=W, V=0.68):
    lmda_f = list(range(350, 810, 10))
    mult = 2*np.pi*c*1.5758*eV
    loss_power = 0
    for i in range(0, len(lmda_f)):
        z = (h*c/(lmda_f[i]*1e-9) - q*V) / (k_B*T)
        y = (h*c/(lmda_f[i]*1e-9)) / (k_B*T)
        converted_power_step = ((a(lmda_f[i]*1e-9, thickness=d, case='TMM')/(lmda_f[i]*1e-9)**4)) * (1/(exp(z)-1) - 1/(exp(y)-1)) *gap_lmda
        loss_power += converted_power_step*mult
    return np.real(loss_power)
print('Radiative recomb Loss =', loss_rad_recomb())


def J(d=W, V=0.68, solar_spectra=solar_atm[70::10]):
    lmda_f = list(range(350, 810, 10))
    J_p = 0
    for i in range(0, len(lmda_f)):
        J_p_step = a(lmda_f[i]*1e-9, thickness=d, case='TMM') * solar_spectra[i] * (lmda_f[i]*1e-9) * gap_lmda
        J_p += J_p_step * (q/(h*c))
    return J_p - loss_rad_recomb(d=d, V=V)/1.5758
print('J =', J())


def loss_spacial_relax(d=W, V=0.68, solar_spectra=solar_atm[70::10]):
    return J(d=d, V=V, solar_spectra=solar_spectra) * (1.5758 - V)
print('Spacial Relaxation Loss =', loss_spacial_relax())



converted_power_atm = [0]*len(lmda)
converted_power_ext = [0]*len(lmda)
for i in range(0, len(lmda)):
    converted_power_atm[i] = absorb_tmm[i]*solar_spectra_atm_lst[i]
    converted_power_ext[i] = absorb_tmm[i]*solar_spectra_ext_lst[i]


conv_power_atm = 0
conv_power_ext = 0
for i in range(0, len(lmda)):
    conv_power_atm += converted_power_atm[i]*gap_lmda
    conv_power_ext += converted_power_ext[i]*gap_lmda

true_converted_power_atm = conv_power_atm - loss_rad_recomb() - loss_spacial_relax() - loss_therm()
true_converted_power_ext = conv_power_ext - loss_rad_recomb() - loss_spacial_relax(solar_spectra=solar_ext[70::10]) - loss_therm(solar_spectra=solar_ext[70::10])
atm_eff = true_converted_power_atm/solar_incident_atm
ext_eff = true_converted_power_ext/solar_incident_ext
print('True Pow. (atm) = ', true_converted_power_atm)
print('True Pow. (ext) = ', true_converted_power_ext)
print('True Eff. (atm) = ', atm_eff)
print('True Eff. (ext) = ', ext_eff)



plt.plot(lmda, converted_power_atm, label='Inside Atmosphere')
plt.plot(lmda_solar, solar_atm[70:570:10], label='Solar Spectrum - Atmospheric')
plt.xlabel('Wavelength(nm)-->')
plt.ylabel('Absorbed Power(W/m2.nm-1)-->')
plt.legend()
plt.title('Power Absorbed by the HIT Solar Cell - Atmospheric; PCE = '+ "{:.2f}".format(atm_eff*100) + '%')
plt.show()

plt.plot(lmda, converted_power_ext, label='Extraterrestrial')
plt.plot(lmda_solar, solar_ext[70:570:10], label='Solar Spectrum - Extraterrestrial')
plt.xlabel('Wavelength(nm)-->')
plt.ylabel('Absorbed Power(W/m2.nm-1)-->')
plt.legend()
plt.title('Power Absorbed by the HIT Solar Cell - Extraterrestrial; PCE = '+ "{:.2f}".format(ext_eff*100) + '%')
plt.show()



def eff_abs(d, env='atm', solar_spectra=solar_atm[70::10], lmda_th=int(lmda_max)):
    lmda_f = list(range(350, 810, 10))
    converted_power = 0
    for i in range(0, len(lmda_f)):
        converted_power += a(lmda_f[i]*1e-9, thickness=d, case='TMM')*solar_spectra[i]*gap_lmda
    
    return converted_power/solar_incident_atm


def conv_pow(d=W, solar_spectra=solar_atm[70::10]):
    lmda_f = list(range(350, 810, 10))
    converted_pow = 0
    for i in range(0, len(lmda_f)):
        converted_pow += a(lmda_f[i]*1e-9, thickness=d, case='TMM')*solar_spectra[i]*gap_lmda
    return converted_pow


def PCE(d=W):
    return (J(d=d)*0.68)/conv_power_atm


thick = np.linspace(1e-9, 1000e-9)
effective_abs = [eff_abs(i, env='atm') for i in thick]
plt.plot(thick, effective_abs, label='')
plt.xscale('log')
plt.xlabel('Perovskite Thickness(m)-->')
plt.ylabel('Effective Absorbtance-->')
# plt.legend()
plt.title('Effective Absorption as a function of Perovskite thickness')
plt.show()

pow_conv_eff = [PCE(d=i) for i in thick]
plt.plot(effective_abs, pow_conv_eff, label='')
# plt.xscale('log')
plt.xlabel('Effective Absobtance-->')
plt.ylabel('Power conversion Efficiency-->')
# plt.legend()
plt.title('Power Conversion Efficiency as a function of Effective Absorptance')
plt.show()
# print(pow_conv_eff)


L_rad_eff = [loss_rad_recomb(d=i)/conv_power_atm for i in thick]
L_therm_eff = [loss_therm(d=i)/conv_power_atm for i in thick]
L_spa_eff = [loss_spacial_relax(d=i)/conv_power_atm for i in thick]
plt.plot(effective_abs, pow_conv_eff, label='Extracted Power')
plt.plot(effective_abs, L_rad_eff, label='Recombination Losses')
plt.plot(effective_abs, L_therm_eff, label='Thermal Losses')
plt.plot(effective_abs, L_spa_eff, label='Spacial Relaxation Loss')
plt.xlabel('Effective Absobtance-->')
plt.ylabel('Losses & Power Conversion Efficiency-->')
plt.legend()
plt.title('Power Conversion Efficiency as a function of Effective Absorptance')
plt.show()

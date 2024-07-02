# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:57:25 2024

@author: a4246
"""

from scipy.interpolate import interp1d
import numpy as np


from tmm.tmm_core import inc_tmm, coh_tmm
from dielectric_models import sellmeier



def N(lam_vac, params):
    """
    Calculation of the complexe refractive index for 

    For the substrate ns and nk are eather calculated with Cauchy or 
    are known values which are selected in a .csv file.
    To have a function out of the values n and k are interpolated.

    For the thin layer n_TL and k_TL are calculated with Tauc-Lorentz.
    """
    _, _, n, k = sellmeier(lam_vac, params)

    N_tmm = n + 1j * k
    return N_tmm

def ema(lam_vac, input_ema):
    N_1, N_2, f1, f2 = input_ema
    if 0<f1<1:
        
        n_1 = np.real(N_1)
        k_1 = np.imag(N_1)
    
        e1_1 = n_1**2 - k_1**2
        e2_1 = 2 * n_1 * k_1
    
        e_1 = e1_1 - 1j * e2_1
    
        n_2 = np.real(N_2)
        k_2 = np.imag(N_2)
    
        e1_2 = n_2**2 - k_2**2
        e2_2 = 2 * n_2 * k_2
    
        e_2 = e1_2 - 1j * e2_2
    
        p = np.sqrt(e_1/e_2)
    
        b = 1/4 * ((3 * f2 - 1) * (1 / p - p) + p)
    
        z = b + np.sqrt(b**2 + 0.5)
    
        e_ema = z * np.sqrt(e_1*e_2)
    
        e1_ema = np.real(e_ema)
        e2_ema = np.imag(e_ema)
    
        n_ema = np.sqrt((e1_ema + (e1_ema**2 + e2_ema**2)**(0.5)) / 2)
    
        k_ema = np.sqrt((-e1_ema + (e1_ema**2 + e2_ema**2)**(0.5)) / 2)
    
        N_ema = n_ema + 1j * k_ema
        
    else:
        N_ema = 100000
    return N_ema

def RT_SM(lam_vac, params, args):

    n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, subs = args

    d_list = [np.inf, params[-1], d_s_RT, np.inf]
    
    
    params_x = np.zeros(len(params)+1)
    params_x[0] = n_osz
    
    params_x[1:] = params

    N_tmm = N(lam_vac, params_x[0:-1])


    n_s = interp1d(subs[0], subs[1].real, kind='linear')(lam_vac) 
    k_s = interp1d(subs[0], subs[1].imag, kind='linear')(lam_vac) 
    
    N_s = n_s + 1j * k_s
    
    n_list = [1, N_tmm, N_s, 1]

    R_s = inc_tmm('s', n_list, d_list, c_list,
                  theta_0_R/180*np.pi, lam_vac)['R']

    R_p = inc_tmm('p', n_list, d_list, c_list,
                  theta_0_R/180*np.pi, lam_vac)['R']

    R = (R_s + R_p)/2

    Ts = inc_tmm('s', n_list, d_list, c_list, theta_0_T/180*np.pi, lam_vac)['T']
    Tp = inc_tmm('p', n_list, d_list, c_list, theta_0_T/180*np.pi, lam_vac)['T']
    T = 0.5 * (Ts +Tp)

    return R, T

def RT_rough_SM(lam_vac, theta, params, args):

    n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, subs = args
    """
    Main Model for calculation of psi and Delta for certain wavelength 
    
    Input variables
    
    lam_vac: wavelength (nm)
    Eg, C, E0, A, eps_1_inf, d_f : Tauc-Lorents parameters and thin layer thickness
    th_0: angle of incidence (rad)
    
    Output
    
    psi: according to ellipsometry measurement (rad)
    Delta: accourding to ellipsometry measurement (rad)
    """

        # list of layer thickness (starting from layer where lights incidently enters.
    d_list = [np.inf, params[-2], params[-3], d_s_RT, np.inf]

    params_x = np.zeros(len(params)+1)
    params_x[0] = n_osz
    
    params_x[1:] = params

    N_tmm = N(lam_vac, params_x[0:-3])

    N_ema = ema(lam_vac, (N_tmm, 1, params[-1], 1-params[-1]))

    n_s = interp1d(subs[0], subs[1].real, kind='linear')(lam_vac) 
    k_s = interp1d(subs[0], subs[1].imag, kind='linear')(lam_vac) 
    
    N_s = n_s + 1j * k_s
    
    n_list = [1, N_ema, N_tmm, N_s, 1]

    R_s = inc_tmm('s', n_list, d_list, c_list,
                  theta_0_R/180*np.pi, lam_vac)['R']

    R_p = inc_tmm('p', n_list, d_list, c_list,
                  theta_0_R/180*np.pi, lam_vac)['R']

    R = (R_s + R_p)/2

    Ts = inc_tmm('s', n_list, d_list, c_list, theta_0_T/180*np.pi, lam_vac)['T']
    Tp = inc_tmm('p', n_list, d_list, c_list, theta_0_T/180*np.pi, lam_vac)['T']
    T = 0.5 * (Ts +Tp)
    return R, T

def SE_SM(lam_vac, theta, params, args):

    """
    Main Model for calculation of psi and Delta for certain wavelength 
    
    Input variables
    
    lam_vac: wavelength (nm)
    Eg, C, E0, A, eps_1_inf, d_f : Tauc-Lorents parameters and thin layer thickness
    th_0: angle of incidence (rad)
    
    Output
    
    psi: according to ellipsometry measurement (rad)
    Delta: accourding to ellipsometry measurement (rad)
    """
    (n_osz, subs) = args
    params_x = np.zeros(len(params)+1)
    params_x[0] = n_osz

    params_x[1:] = params

    # list of layer thickness (starting from layer where lights incidently enters.
    d_list = [np.inf, params[-1], np.inf]

    # calculation of complex refractive index
    N_tmm = N(lam_vac, params_x[0:-1])
    
    n_s = interp1d(subs[0], subs[1].real, kind='linear')(lam_vac) 
    k_s = interp1d(subs[0], subs[1].imag, kind='linear')(lam_vac) 
    
    N_s = n_s + 1j * k_s

    # list of refractive index, same order as d_list
    n_list = [1, N_tmm, N_s]

    # calculation of reflactance amplitude for coherent layers
    rs = coh_tmm('s', n_list, d_list, theta/180*np.pi, lam_vac)['r']
    # calculation of reflactance amplitude for coherent layers
    rp = coh_tmm('p', n_list, d_list, theta/180*np.pi, lam_vac)['r']

    psi = np.arctan(np.abs(rp/rs))
    delta = -1 * np.angle(-rp/rs) + np.pi
    return psi, delta

def SE_rough_SM(lam_vac, theta, params, args):

    """
    Main Model for calculation of psi and Delta for certain wavelength 
    
    Input variables
    
    lam_vac: wavelength (nm)
    Eg, C, E0, A, eps_1_inf, d_f : Tauc-Lorents parameters and thin layer thickness
    th_0: angle of incidence (rad)
    
    Output
    
    psi: according to ellipsometry measurement (rad)
    Delta: accourding to ellipsometry measurement (rad)
    """
    (n_osz, subs) = args
    params_x = np.zeros(len(params)+1)
    params_x[0] = n_osz

    params_x[1:] = params

    # list of layer thickness (starting from layer where lights incidently enters.
    d_list = [np.inf, params[-2], params[-3], np.inf]

    # calculation of complex refractive index
    N_tmm = N(lam_vac, params_x[0:-3])
    n_s = interp1d(subs[0], subs[1].real, kind='linear')(lam_vac) 
    k_s = interp1d(subs[0], subs[1].imag, kind='linear')(lam_vac) 
    
    N_s = n_s + 1j * k_s

    N_r1 = ema(lam_vac, (N_tmm, 1, params[-1], 1-params[-1]))

    # list of refractive index, same order as d_list
    n_list = [1, N_r1, N_tmm, N_s]

    # calculation of reflactance amplitude for coherent layers
    rs = coh_tmm('s', n_list, d_list, theta/180*np.pi, lam_vac)['r']
    # calculation of reflactance amplitude for coherent layers
    rp = coh_tmm('p', n_list, d_list, theta/180*np.pi, lam_vac)['r']

    psi = np.arctan(np.abs(rp/rs))
    delta = -1 * np.angle(-rp/rs) + np.pi
    return psi, delta
 
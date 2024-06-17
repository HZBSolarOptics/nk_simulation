# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:57:25 2024

@author: a4246
"""

from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


import numpy as np

import load_data as ld

from tmm.tmm_core import inc_tmm, coh_tmm
from dielectric_models import TL_multi, TL_drude


def N(lam_vac, params):
    """
    Calculation of the complexe refractive index for 

    For the substrate ns and nk are eather calculated with Cauchy or 
    are known values which are selected in a .csv file.
    To have a function out of the values n and k are interpolated.

    For the thin layer n_TL and k_TL are calculated with Tauc-Lorentz.
    """
    _, _, n, k = TL_drude(lam_vac, params)

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


def RT(lam_vac, params, args):

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
def RT_rough(lam_vac, params, args):

    n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, subs = args

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


def SE(lam_vac, theta, params, args):
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

def SE_rough(lam_vac, theta, params, args):


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


def model_3(lam_vac, theta, params, args):

    if any(param < 0 for param in params):
        R = 1e10
        T = 1e10
        psi = 1e10
        delta = 1e10
    else:    
        (n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, subs) = args
    
        d_list_RT = [np.inf, params[-1], params[-2], d_s_RT, np.inf]
    
        params_x = np.zeros(len(params)+1)
        params_x[0] = n_osz
        params_x[1:] = params
    
        N_tmm = N(lam_vac, params_x[0:-3])
        N_r = ema(lam_vac, (N_tmm, 1, params[-3], 1-params[-3]))
    
        index = np.where(subs[:, 0] == lam_vac)
        N_glass = np.squeeze(subs[index, 1])
    
        n_RT_list = [1, N_r, N_tmm, N_glass, 1]
    
        R_s = inc_tmm('s', n_RT_list, d_list_RT, c_list,
                      theta_0_R/180*np.pi, lam_vac)['R']
        T_s = inc_tmm('s', n_RT_list, d_list_RT, c_list, 0, lam_vac)['T']
        R_p = inc_tmm('p', n_RT_list, d_list_RT, c_list,
                      theta_0_R/180*np.pi, lam_vac)['R']
        T_p = inc_tmm('p', n_RT_list, d_list_RT, c_list, 0, lam_vac)['T']
    
        R = 0.5 * (R_s + R_p)
        T = 0.5 * (T_s + T_p)
    
        n_ell_list = [1, N_r, N_tmm, N_glass, 1]
        d_list_ell = [np.inf, params[-1], params[-2], d_s_RT, np.inf]
    
        # calculation of reflactance amplitude for coherent layers
        rs = coh_tmm('s', n_ell_list, d_list_ell, theta/180*np.pi, lam_vac)['r']
        # calculation of reflactance amplitude for coherent layers
        rp = coh_tmm('p', n_ell_list, d_list_ell, theta/180*np.pi, lam_vac)['r']
    
        psi = np.arctan(np.abs(rp/rs))
        delta = -1 * np.angle(-rp/rs) + np.pi
        
    return R, T, psi, delta


def error_RT(params, *args):

    n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, lam_vac_RT, R_data, T_data, subs, alpha_R, alpha_T = args

    model_values = np.array([RT(lam_vac, params, (n_osz, d_s_RT,
                                                     theta_0_T, theta_0_R,
                                                     c_list, subs)) for lam_vac in lam_vac_RT]).T
    R_values, T_values = model_values[0], model_values[1]

    max_R = R_data.max()
    max_T = T_data.max()
    
    Error_R = (R_values - R_data) * 100 / max_R / np.sqrt(len(R_data))
    Error_T = (T_values - T_data) * 100 / max_T / np.sqrt(len(T_data))

    MSE = np.sqrt(((R_values - R_data)**2 + (T_values - T_data)**2).mean()) * 100
    
    plot_RT(MSE, alpha_R, alpha_T, lam_vac_RT,
            R_values, T_values, R_data, T_data)
    print('Parameters:',( ', '.join([format(param, '.4f') for param in params]), MSE))
    return np.concatenate((alpha_R*Error_R,
                           alpha_T*Error_T))


def error_SE(params, *args):
    """
    Calculation of the squared Error between the calculatet and experimental values of psi and Delta.

    Input

    params: Tauc-Lorents parameters (eV) and film thickness (nm)

    Ouput

    total Error of psi and Delta in complete wavelength range
    """
    (n_osz, subs,
     theta_values, lam_vac_SE, psis_data, deltas_data,
     alpha_psi, alpha_delta) = args

    psis = []
    deltas = []

    for theta in theta_values:

        psi_values, delta_values = np.array([SE_rough(lam_vac, theta, params, (n_osz, subs)) for lam_vac in lam_vac_SE]).T
        psis.append(psi_values)
        deltas.append(delta_values)

    
    psis_fl= np.array(psis).flatten()
    deltas_fl= np.array(deltas).flatten()

    Errors_psi = (psis_fl - psis_data)/ np.sqrt(len(psis_data))
    Errors_delta = (deltas_fl - deltas_data) / np.sqrt(len(deltas_data))
    
    threshold = 1
    indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
    print(indices)
    # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
    indices_to_remove = []
    
    # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
    for index in indices:
        indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
    
    # Entferne Duplikate und sortiere die Liste der Indizes
    indices_to_remove = sorted(list(set(indices_to_remove)))

    # Entferne die entsprechenden Messwerte und Fehler
    
    Errors_psi_filtered = np.delete(Errors_psi, indices_to_remove)
    Errors_delta_filtered = np.delete(Errors_delta, indices_to_remove)
    
    # Aktualisiere max_psi und max_delta
    psi_data_filtered = np.delete(psis_data, indices_to_remove)
    deltas_data_filtered = np.delete(deltas_data, indices_to_remove)
    max_psi = psi_data_filtered.max()
    max_delta = deltas_data_filtered.max()
    
    # Berechne die Fehler ohne die entfernten Messwerte
    Error_psi_filtered = Errors_psi_filtered / max_psi * 100 
    Error_delta_filtered = Errors_delta_filtered / max_delta * 100 
    
    # Berechne die MSE ohne die entfernten Messwerte
    MSE_SE_filtered = np.sqrt(np.sum((alpha_psi * Errors_psi_filtered)**2) +np.sum((alpha_delta * Errors_delta_filtered)**2)) * 180 / np.pi
    
    plot_SE(MSE_SE_filtered, MSE_SE_filtered, theta_values, lam_vac_SE, psis_fl, deltas_fl, psis_data, deltas_data)
    print('Parameters:', (', '.join([format(param, '.4f') for param in params]), MSE_SE_filtered))
    return np.concatenate((alpha_psi*Error_psi_filtered,
                           alpha_delta*Error_delta_filtered))


def error_tot(params, *args):

    (n_osz, d_s_RT,
     theta_0_T, theta_0_R,
     c_list,
     lam_vac_RT, R_data, T_data,
     subs,
     theta_values,
     lam_vac_SE, psis_data, deltas_data,
     alpha_R, alpha_T, alpha_psi, alpha_delta) = args

    psis = []
    deltas = []

    for theta in theta_values:

       psi_values, delta_values = np.array([SE_rough(lam_vac, theta, params, (n_osz, subs)) for lam_vac in lam_vac_SE]).T
       psis.append(psi_values)
       deltas.append(delta_values)
        
    R_values, T_values = np.array([RT_rough(lam_vac, params, (n_osz, d_s_RT,
                                                     theta_0_T, theta_0_R,
                                                     c_list, subs)) for lam_vac in lam_vac_SE]).T
        
    max_R = R_data.max()
    max_T = T_data.max()
    
    Error_R = (R_values - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_SE))
    Error_T = (T_values - T_data) * 100 / max_T / np.sqrt(len(lam_vac_SE))
    
    MSE_RT = np.sqrt(((R_values - R_data)**2 + (T_values - T_data)**2).mean()) * 100

    
    psis_fl= np.array(psis).flatten()
    deltas_fl= np.array(deltas).flatten()
    
    Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
    Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
    
    threshold = 1

    # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
    indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
   
    # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
    indices_to_remove = []
    
    # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
    for index in indices:
        indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
    
    # Entferne Duplikate und sortiere die Liste der Indizes
    indices_to_remove = sorted(list(set(indices_to_remove)))
    print(indices_to_remove)

    # Entferne die entsprechenden Messwerte und Fehler
    
    Errors_delta_filtered = np.delete(Errors_delta, indices_to_remove)
    
    # Aktualisiere max_psi und max_delta
    deltas_data_filtered = np.delete(deltas_data, indices_to_remove)
    max_psi = psis_data.max()
    max_delta = deltas_data_filtered.max()
    
    # Berechne die Fehler ohne die entfernten Messwerte
    Error_delta_filtered = Errors_delta_filtered / max_delta * 100 
    Error_psi = Errors_psi / max_psi * 100
    # Berechne die MSE ohne die entfernten Messwerte
    MSE_SE_filtered = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
    
    # Berechne die summierte MSE ohne die entfernten Messwerte

    plot_tot(MSE_RT, MSE_SE_filtered, R_values, T_values, lam_vac_SE, R_data,
             T_data, theta_values, psis_fl, deltas_fl, psis_data, deltas_data)
    
    print('Parameters:', (', '.join([format(param, '.4f') for param in params]), MSE_RT, MSE_SE_filtered))
    return np.concatenate((alpha_R*Error_R,
                          alpha_T*Error_T,
                          alpha_psi*Error_psi,
                          alpha_delta*Error_delta_filtered))


def plot_RT(MSE, alpha_R, alpha_T, lam_vac_RT, R, T, R_data, T_data):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.text(0.98, 0.98, f'MSE={MSE:.2f} %', transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.plot(lam_vac_RT, T*100, label='$T$', color='black', linewidth=0.7)
    ax.plot(lam_vac_RT, T_data*100, color='black',linestyle='--')
    ax.plot(lam_vac_RT, R*100, label='$R$', color='blue', linewidth=0.7)
    ax.plot(lam_vac_RT, R_data*100,  color='blue', linestyle='--')
    ax.set_title('Transfer-Matrix-Method RT', fontsize=14)
    ax.set_ylabel('R/T', fontsize=14)
    ax.set_xlabel('$\lambda [nm]$', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=14)
    plt.show()


def plot_SE(MSE_psi, MSE_delta, theta_values, lam_vac, psi, delta, psi_data, delta_data):
    psi = np.array(psi).T
    delta = np.array(delta).T
    fig, axs = plt.subplots(1,2, figsize=(8, 6))
    colors = ['black', 'green', 'purple']
    for i, theta in enumerate(theta_values):
        
        start = 0 + i * len(lam_vac)
        end = start +  len(lam_vac)
        
        axs[0].plot(lam_vac, psi[start:end]*180/np.pi,
                    label=f'${theta}°$', color=colors[i], linewidth=0.7)
        axs[0].plot(lam_vac, psi_data[start:end]*180/np.pi, color=colors[i],
                    linestyle='--', linewidth=0.7)
        
        axs[1].plot(lam_vac, delta[start:end]*180/np.pi, color=colors[i], linewidth=0.7)
        axs[1].plot(lam_vac, delta_data[start:end]*180/np.pi, color=colors[i],
                    linestyle='--', linewidth=0.7 )

    axs[1].text(0.98, 0.98, f'RMSE={MSE_psi:.2f} °', transform=axs[1].transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axs[1].set_xlabel('$\lambda [nm]$', fontsize=14)
    axs[1].set_ylabel('$\psi$ [°]', fontsize=14)
    axs[0].set_ylabel('$\Delta$ [°]', fontsize=14)
    
    plt.show()



def plot_tot(MSE_RT, MSE_SE, R, T, lam_vac, R_data, T_data, theta_values, psi, delta, psi_data, delta_data):
    # Plot für RT
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot für RT
    axs[0].text(0.98, 0.98, f'RMSE={MSE_RT:.2f} %', transform=axs[0].transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axs[0].plot(lam_vac, T*100, label='$T$', color='black', linewidth=0.7)
    axs[0].plot(lam_vac, T_data*100, color='black',
                linestyle='--')
    axs[0].plot(lam_vac, R*100, label='$R$', color='blue', linewidth=0.7)
    axs[0].plot(lam_vac, R_data*100,  color='blue', linestyle='--')
    axs[0].set_title('Transfer-Matrix-Method RT', fontsize=14)
    axs[0].set_ylabel('R/T', fontsize=14)
    axs[0].set_xlabel('$\lambda [nm]$', fontsize=14)
    axs[0].legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=14)

    # Plot für psi und delta
    colors = ['black', 'green', 'purple']
    axs3 = axs[1].twinx()
    for i, theta in enumerate(theta_values):
        start = 0 + i * len(lam_vac)
        end = start +  len(lam_vac)
        axs[1].plot(lam_vac, psi[start:end]*180/np.pi,
                    label=f'${theta}°$', color=colors[i], linewidth=0.7)
        axs[1].plot(lam_vac, psi_data[start:end]*180/np.pi, color=colors[i],
                     markersize=1.5, linestyle='--')
        axs3.plot(lam_vac, delta[start:end]*180/np.pi,
                     color=colors[i], linewidth=0.7)
        axs3.plot(lam_vac, delta_data[start:end]*180/np.pi, color=colors[i],
                     linestyle='--')

    axs[1].text(0.98, 0.98, f'RMSE={MSE_SE:.2f} °', transform=axs[1].transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axs[1].set_xlabel('$\lambda [nm]$', fontsize=14)
    axs[1].set_ylabel('$\psi$/$\Delta$', fontsize=14)
    axs[1].set_title('Transfer-Matrix-Method $\psi$ and $\Delta$', fontsize=14)
    axs[1].legend(loc='lower right', bbox_to_anchor=(0.98, 0.02), fontsize=14)

    plt.tight_layout()
    plt.show()


"""
Upload of nk data from ellipsometry fit, ellipsometry measurement and substrate nk data


"""

def load(minimal, maximal):
    lam_vacsubs_RT_values, M_RT_real, M_RT_imag = ld.load_nksubsdata()
    lam_vac_RT, R_data, T_data = ld.load_RT(minimal, maximal)
  
   
    theta_values,lam_vac_SE, psis_data, deltas_data = ld.load_SE(
        minimal, maximal)
    
    return (lam_vacsubs_RT_values, M_RT_real, M_RT_imag,
            lam_vac_RT, R_data, T_data,
            theta_values, lam_vac_SE, psis_data, deltas_data)

def main():
    """
    In the main function the optimization is done using the minimize function for minimizing the error.

    The results are calculated using the optimized Tauc-Lorentz parameters and the film thickness.
    The results are shown in graphics.
    """
    
    sample = '4617'
    n_osz = 1
    #theta_values = [50,60,70]
    minimal, maximal = ld.lambda_range()

    (lam_vacsubs_values, M_real, M_imag,
     lam_vac_RT, R_data, T_data,
     theta_values,lam_vac_SE, psis_data, deltas_data) = load(minimal, maximal)
    
    print(np.shape(psis_data))
    print(theta_values)
    T_data_reg = interp1d(lam_vac_RT, T_data, kind='linear')(lam_vac_SE)
    R_data_reg = interp1d(lam_vac_RT, R_data, kind='linear')(lam_vac_SE)

    n_s = interp1d(lam_vacsubs_values, M_real[:, 1], kind='linear')(lam_vac_RT)
    k_s = interp1d(lam_vacsubs_values, M_imag[:, 1], kind='linear')(lam_vac_RT)

    Ns_RT = n_s + 1j * k_s

    n_s2 = interp1d(lam_vacsubs_values,
                    M_real[:, 1], kind='linear')(lam_vac_SE)
    k_s2 = interp1d(lam_vacsubs_values,
                    M_imag[:, 1], kind='linear')(lam_vac_SE)

    Ns_SE = n_s2 + 1j * k_s2

    subs_RT = np.array([lam_vac_RT, Ns_RT]).T
    subs_SE = np.array([lam_vac_SE, Ns_SE]).T
    psis_data = np.array(psis_data)
    deltas_data = np.array(deltas_data)

    # Materialkonstanten
    d_s_RT = 1.1e6

    theta_0_R = 8
    theta_0_T = 0

    alpha_R = 0
    alpha_T = np.zeros(len(lam_vac_SE))
    index = np.where(lam_vac_SE > 700)[0]
    
    alpha_T[index[0]:] = np.full(len(lam_vac_SE)-index[0], 3)
    
    alpha_psi = 1
    alpha_delta = 1

    c_list = ['i', 'c', 'c', 'i', 'i']
    
    args = (n_osz, d_s_RT,
            theta_0_T, theta_0_R,
            c_list,
            lam_vac_RT,
            R_data_reg, T_data_reg, subs_SE,
            theta_values, lam_vac_SE, psis_data.flatten(), deltas_data.flatten(),
            alpha_R, alpha_T, alpha_psi, alpha_delta)

    args_RT = (n_osz, d_s_RT,
               theta_0_T, theta_0_R,
               c_list,
               lam_vac_RT, R_data, T_data,
               subs_RT,
               alpha_R, alpha_T)

    args_SE = (n_osz, subs_SE,
               theta_values, lam_vac_SE, psis_data.flatten(), deltas_data.flatten(),
               alpha_psi, alpha_delta)

    initial_guess = np.array([2.7,0.2,0.0,1.9,2.6,3.9,87.9,0.6,211.7,11.1])

    # opt_param = least_squares(
    #     error_RT, initial_guess, args=args_RT,  verbose=2, method = 'lm', max_nfev=100000) 
    # opt_param = least_squares(
    #     error_SE, initial_guess, args=args_SE,  verbose=2, method = 'lm', max_nfev=100000) 
    opt_param = least_squares(
        error_tot, initial_guess, args=args,  verbose=2, method = 'lm', max_nfev=100000) 

    print(opt_param)

    params_x = np.zeros(len(opt_param.x)+1)
    params_x[0] = n_osz
    params_x[1:-3] = opt_param.x[0:-3]

    print('Parameters:', ', '.join([format(param, '.6f') for param in opt_param.x]))
    values = [TL_drude(632, params_x)[2:4], TL_drude(800, params_x)[2:4]]

    # Formatiere und gib die Ergebnisse aus
    print("nk(632 nm, 800 nm):", ','.join([format(value, '.6f') for value in np.array(values).flatten()]))
    
if __name__ == "__main__":
    main()

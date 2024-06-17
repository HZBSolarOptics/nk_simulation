# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:57:25 2024

@author: a4246
"""

from scipy.optimize import minimize, shgo, differential_evolution
from scipy.interpolate import interp1d


import numpy as np

import load_data as ld

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
 
def error_RT(params, *args):
    (eps_1_inf, B1, B2, C1, C2, d_f) = params
    d_s_RT, theta_0_T, theta_0_R, c_list, n_3, k_3, lam_vac_RT, R_data, T_data, lam_vacsubs_values, M_real, M_imag, alpha_R, alpha_T  = args
    model_values = np.array([model(lam_vac, params, args) for lam_vac in lam_vac_RT]).T
    R_tmm_values, T_tmm_values = model_values[0], model_values[1]
    
    N_lambdaRT = len(lam_vac_RT)
    Error_R = np.sqrt(1/N_lambdaRT * np.sum((R_tmm_values - R_data)**2))
    Error_T = np.sqrt(1/N_lambdaRT * np.sum((T_tmm_values - T_data)**2))
    return alpha_R * Error_R + alpha_T * Error_T

def error_ellips(params, *args):
    """
    Calculation of the squared Error between the calculatet and experimental values of psi and Delta.
    
    Input
    
    params: Tauc-Lorents parameters (eV) and film thickness (nm)
    
    Ouput
    
    total Error of psi and Delta in complete wavelength range
    """
    (eps_1_inf, B1, B2, C1, C2, d_f) = params
    n_3, k_3, lam_vacsubs_values, M_real, M_imag, theta_values, lam_vac_psidelta, psis_data, deltas_data, alpha_psi, alpha_delta = args
    Error_psi, Error_delta = 0, 0
    
    for theta, psi_data, delta_data in zip(theta_values, psis_data, deltas_data):

        model_2_values = np.array([model_2(lam_vac, theta, params, args) for lam_vac in lam_vac_psidelta]).T
        psi_values, delta_values = model_2_values[0], model_2_values[1]

        N_lambda = len(lam_vac_psidelta)
        

        psi_error = 1/3 * np.sqrt(1 / N_lambda * np.sum((psi_values - psi_data)**2))
        delta_error = 1/3 * np.sqrt(1 / N_lambda * np.sum((delta_values - delta_data)**2))
        
        Error_psi += psi_error
        Error_delta += delta_error

    return alpha_psi * Error_psi + alpha_delta  * Error_delta

def error_tot(params, *args):
    (a, b, d_f) = params
    
    (d_s_RT, 
     theta_0_T, theta_0_R, 
     c_list, 
     n_3, k_3, 
     lam_vac_RT, R_data, T_data, 
     lam_vacsubs_RT_values, M_RT_real, M_RT_imag, 
     lam_vacsubs_ell_values, M_ell_real, M_ell_imag,
     theta_values, 
     lam_vac_psidelta, psis_data, deltas_data, 
     alpha_R, alpha_T, alpha_psi, alpha_delta)  = args
    

    Error_R, Error_T, Error_psi, Error_delta = 0, 0, 0, 0
    
    for theta, psi_data, delta_data in zip(theta_values, psis_data, deltas_data):

        model_3_values = np.array([model_3(lam_vac, theta, params, args) for lam_vac in lam_vac_psidelta]).T
        R_values, T_values = model_3_values[1], model_3_values[2]
        psi_values, delta_values = model_3_values[2], model_3_values[3]
    
        N_lambda = len(lam_vac_psidelta)
        
        R_error = 1/3 * np.sqrt(1 / N_lambda * np.sum((R_values - R_data)**2))
        T_error = 1/3 * np.sqrt(1 / N_lambda * np.sum((T_values - T_data)**2))
        psi_error = 1/3 * np.sqrt(1 / N_lambda * np.sum((psi_values - psi_data)**2))
        delta_error = 1/3 * np.sqrt(1 / N_lambda * np.sum((delta_values - delta_data)**2))
        
        Error_R += R_error
        Error_T += T_error
        Error_psi += psi_error
        Error_delta += delta_error
        print(theta, psi_error, delta_error)
    return alpha_R * Error_R + alpha_T * Error_T + alpha_psi * Error_psi + alpha_delta * Error_delta
"""
Upload of nk data from ellipsometry fit, ellipsometry measurement and substrate nk data


"""   
def load(theta_values, minimal, maximal):
    lam_vacsubs_RT_values, M_RT_real, M_RT_imag = ld.load_nksubsdata()
    lam_vac_values, R_data, T_data = ld.load_data(minimal, maximal)   
   
    return (lam_vacsubs_RT_values, M_RT_real, M_RT_imag, 
            lam_vac_values, R_data, T_data)


def main():
    
    """
    In the main function the optimization is done using the minimize function for minimizing the error.
    
    The results are calculated using the optimized Tauc-Lorentz parameters and the film thickness.
    The results are shown in graphics.
    """
    n_3 = 1
    k_3 = 0
    #Materialkonstanten
    d_s_RT = 1.1e6
    # d_s_el = 1.1e6

    theta_0_R = 8
    theta_0_T = 8

    alpha_R = 0.5
    alpha_T = 0.5
    alpha_psi = 0.25
    alpha_delta = 0.25

    
    c_list = ['i','c','i','i']
    
    theta_values = [50, 60, 70]
    minimal, maximal = ld.lambda_range()
    
    (lam_vacsubs_RT_values, M_RT_real, M_RT_imag, 
    lam_vacsubs_ell_values, M_ell_real, M_ell_imag, 
    lam_vac_RT, R_data, T_data, 
    lam_vac_psidelta, psis_data, deltas_data) = load(theta_values, minimal, maximal)
    
   
    
    last_successful_result = None

    def progress_callback(xk):
        nonlocal  last_successful_result

        if last_successful_result is not None:
            print("Letztes erfolgreiches Ergebnis:")
            print(last_successful_result)
        raise StopIteration
        
        last_successful_result = xk
    
    args = (d_s_RT, 
            theta_0_T, theta_0_R, 
            c_list, 
            n_3, k_3, 
            lam_vac_RT, 
            R_data, T_data, 
            lam_vacsubs_RT_values, M_RT_real, M_RT_imag, 
            lam_vacsubs_ell_values, M_ell_real, M_ell_imag, 
            theta_values, lam_vac_psidelta, psis_data, deltas_data, 
            alpha_R, alpha_T, alpha_psi, alpha_delta)
    
    args_RT = (d_s_RT, 
               theta_0_T, theta_0_R, 
               c_list, n_3, k_3, 
               lam_vac_RT, R_data, T_data, 
               lam_vacsubs_RT_values, M_RT_real, M_RT_imag, 
               alpha_R, alpha_T)
    
    args_ell= (n_3, k_3, 
               lam_vacsubs_ell_values, M_ell_real, M_ell_imag, 
               theta_values, lam_vac_psidelta, psis_data, deltas_data, 
               alpha_psi, alpha_delta)
    
    
    # initial_guess = [1, 1, 1, 50, 1, 200]
    bounds = [(1e-3, 10), (1e-3, 10), (1e-3, 10), (1e-3, 10), (1e-3, 10), (190, 250)]
    # opt_param = minimize(error_RT, initial_guess, args = args_RT, options={'disp': True})
    opt_param = differential_evolution(error_RT, bounds, args = args_RT)

    eps_1_inf_opt, B1_opt, B2_opt, C1_opt, C2_opt, d_f_opt = opt_param.x
    param =  eps_1_inf_opt, B1_opt, B2_opt, C1_opt, C2_opt
    #results 
    
    ER = error_RT(opt_param.x, *args_RT)*100
    
    print("ER", ER)
    print('Parameters', list(opt_param.x))
    print("n(632 nm)", sellmeier(632, *param)[2])


if __name__ == "__main__":
     main()

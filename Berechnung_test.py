from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import numpy as np

import load_data as ld

from tmm.tmm_core import coh_tmm, inc_tmm

from dielectric_models import TL_multi, TL_drude, cauchy, sellmeier


def N(lam_vac, params, Model):
    """
    Calculation of the complexe refractive index for 
    
    For the substrate ns and nk are eather calculated with Cauchy or 
    are known values which are selected in a .csv file.
    To have a function out of the values n and k are interpolated.
    
    For the thin layer n_TL and k_TL are calculated with Tauc-Lorentz.
    """
    
    if Model == 'Tauc Lorentz + Drude':
        _, _, n, k = TL_drude(lam_vac, params)
    if Model == 'Tauc Lorentz' :
        _, _, n, k = TL_multi(lam_vac, params)
    if Model == 'Cauchy' :
        _, _, n, k = cauchy(lam_vac, params)

    if Model == 'Sellmeier' :
        _, _, n, k = sellmeier(lam_vac, params)

    N_tmm = n + 1j * k
    
    return N_tmm

def ema(lam_vac, input_ema):
    N_1, N_2, f1, f2 = input_ema
    
    
    n_1 = np.real(N_1)
    k_1 = np.imag(N_1)
    
    e1_1 = n_1**2 - k_1**2
    e2_1 =2 * n_1 * k_1
    
    e_1 = e1_1 - 1j * e2_1
    
    n_2 = np.real(N_2)
    k_2 = np.imag(N_2)
    
    e1_2 = n_2**2 - k_2**2
    e2_2 =2 * n_2 * k_2
    
    e_2 = e1_2 - 1j * e2_2
    
    p = np.sqrt(e_1/e_2)
    
    b = 1/4 * ((3 * f2 - 1) * (1 / p -p) + p)
    
    z = b + np.sqrt(b**2 + 0.5)

    e_ema = z * np.sqrt(e_1*e_2)

    e1_ema = np.real(e_ema)
    e2_ema = np.imag(e_ema)
    
    n_ema = np.sqrt((e1_ema + (e1_ema**2 + e2_ema**2)**(0.5)) / 2)
    
    k_ema = np.sqrt((-e1_ema + (e1_ema**2 + e2_ema**2)**(0.5)) / 2) 
    
    N_ema = n_ema + 1j * k_ema
    return N_ema

def model(lam_vac, params, args):
    Model = args['Model']
    n_osz = args['n']
    subs = args['subs_RT']

    d_s_RT = args['d_s_RT']
    c_list = args['c_list']
    theta_0_R = args['theta_0_R']
    theta_0_T = args['theta_0_T']
    EMA = args['EMA']
    params_x = np.zeros(len(params)+1)
    params_x[0] = n_osz
    params_x[1:] = params
    
    n_s = interp1d(subs[0], subs[1].real, kind='linear')(lam_vac) 
    k_s = interp1d(subs[0], subs[1].imag, kind='linear')(lam_vac) 
    
    N_s = n_s + 1j * k_s
    if EMA == 'true':
        N_tmm = N(lam_vac, params_x[0:-3], Model)

        d_list = [np.inf, params[-2], params[-3], d_s_RT, np.inf]
        N_ema = ema(lam_vac, (N_tmm, 1, params[-1], 1-params[-1]))
        n_list = [1, N_ema, N_tmm, N_s, 1]
        
    elif EMA == 'false':
      
        N_tmm = N(lam_vac, params_x[0:-1], Model)
        d_list = [np.inf, params[-1], d_s_RT, np.inf]
        n_list = [1, N_tmm, N_s, 1]

    R_s = inc_tmm('s', n_list, d_list, c_list,
                  theta_0_R/180*np.pi, lam_vac)['R']

    R_p = inc_tmm('p', n_list, d_list, c_list,
                  theta_0_R/180*np.pi, lam_vac)['R']

    R = (R_s + R_p)/2
    
    T_s = inc_tmm('s', n_list, d_list, c_list,
                  theta_0_T/180*np.pi, lam_vac)['T']
    
    T_p = inc_tmm('p', n_list, d_list, c_list,
                  theta_0_T/180*np.pi, lam_vac)['T']
    T = (T_s + T_p)/2
        
    return  R, T, N_tmm.real, N_tmm.imag


def model_2(lam_vac, params, args, theta):
    
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
    n_osz = args['n']
    EMA = args['EMA']

    subs = args['subs_SE']
    Model = args['Model']
    n_s = interp1d(subs[0], subs[1].real, kind='linear')(lam_vac) 
    k_s = interp1d(subs[0], subs[1].imag, kind='linear')(lam_vac) 
    
    N_s = n_s + 1j * k_s
    
    params_x= np.zeros(len(params)+1)
    params_x[0] = n_osz
    params_x[1:] = params
    
    if EMA == 'true':
        N_tmm = N(lam_vac, params_x[0:-3], Model)

        d_list = [np.inf, params[-2], params[-3], np.inf]
        N_ema = ema(lam_vac, (N_tmm, 1, params[-1], 1-params[-1]))
        n_list = [1, N_ema, N_tmm, N_s]
        
    elif EMA == 'false':
      
        N_tmm = N(lam_vac, params_x[0:-1], Model)
        d_list = [np.inf, params[-1], np.inf]
        n_list = [1, N_tmm, N_s]
       
    
    rs = coh_tmm('s', n_list, d_list, theta/180*np.pi, lam_vac)['r'] # calculation of reflactance amplitude for coherent layers
    rp = coh_tmm('p', n_list, d_list, theta/180*np.pi, lam_vac)['r'] # calculation of reflactance amplitude for coherent layers
        
    psi = np.arctan(np.abs(rp/rs))
    delta = -1 * np.angle(-rp/rs) + np.pi
    
    return psi, delta, N_tmm.real, N_tmm.imag

def error_RT(params, args):
    
 
    lam_vac_examp = args['lam_vac_examp']
    R_data = args['R_data_reg']
    T_data = args['T_data_reg']
    alpha_R = args['alpha_R']
    alpha_T = args['alpha_T']
    
    model_values = np.array([model(lam_vac, params, args) for lam_vac in lam_vac_examp]).T
    R_values, T_values = model_values[0], model_values[1]
    
    Error_R = np.abs((R_values - R_data))*100
    Error_T =  np.abs((T_values - T_data))*100
    MSE_R = np.sum(np.sqrt(np.ones(len(R_data)) + (alpha_R*Error_R)**2) -1) / (len(R_data))
    MSE_T = np.sum(np.sqrt(np.ones(len(T_data)) + (alpha_T*Error_T)**2) -1) / (len(T_data))
    return MSE_R, MSE_T

def error_ellips(params, args):
    """
    Calculation of the squared Error between the calculatet and experimental values of psi and Delta.
    
    Input
    
    params: Tauc-Lorents parameters (eV) and film thickness (nm)
    
    Ouput
    
    total Error of psi and Delta in complete wavelength range
    """

    theta_values = args['theta_values']
    lam_vac_examp =  args['lam_vac_examp']
    psis_data = args['psis_data']
    deltas_data = args['deltas_data']
    alpha_psi =  args['alpha_psi']
    alpha_delta = args['alpha_delta']

    Error_psi, Error_delta = 0, 0

    psis = []
    deltas = []
    for theta in theta_values:
        
        model_2_values = np.array([model_2(lam_vac, params, args, theta) for lam_vac in lam_vac_examp]).T
        psi_values, delta_values = model_2_values[0], model_2_values[1]
        psis.append(psi_values)
        deltas.append(delta_values)
        
  
    psis_data = np.array(psis_data).flatten()
    deltas_data = np.array(deltas_data).flatten()
    
    psis= np.array(psis).flatten()
    deltas= np.array(deltas).flatten()
    
    Errors_psi = np.abs((psis - psis_data))
    Errors_delta = np.abs((deltas - deltas_data))
    Error_psi = Errors_psi * 180 / np.pi 
    Error_delta = Errors_delta* 180 / np.pi

    

    MSE_psi = np.sum(np.sqrt(np.ones(len(psis_data)) + (alpha_psi*Error_psi)**2) -1) / (len(psis_data))
    MSE_delta =np.sum(np.sqrt(np.ones(len(deltas_data)) + (alpha_delta*Error_delta)**2) -1) / (len(deltas_data))
    return MSE_psi, MSE_delta

def error_tot(params, args):
    
    R_data = args['R_data_reg']
    T_data = args['T_data_reg']
    alpha_R = args['alpha_R']
    alpha_T = args['alpha_T']
    
    theta_values = args['theta_values']
    lam_vac_SE =  args['lam_vac_psidelta']
    psis_data = args['psis_data']
    deltas_data = args['deltas_data']
    alpha_psi =  args['alpha_psi']
    alpha_delta = args['alpha_delta']
    
    model_values = np.array([model(lam_vac, params, args) for lam_vac in lam_vac_SE]).T
    R_values, T_values = model_values[0], model_values[1]
   
    psis = []
    deltas = []
    for theta in theta_values:
        
        model_2_values = np.array([model_2(lam_vac, params, args, theta) for lam_vac in lam_vac_SE]).T
        psi_values, delta_values = model_2_values[0], model_2_values[1]
        psis.append(psi_values)
        deltas.append(delta_values)
    
    Error_R = np.abs((R_values - R_data))/(np.sqrt(len(R_data)))*100
    Error_T =  np.abs((T_values - T_data))/(np.sqrt(len(R_data)))*100
    MSE_RT = 0.5*np.sum((alpha_T*Error_T)**2) + 0.5*np.sum((alpha_R*Error_R)**2)
    psis_data = np.array(psis_data).T
    deltas_data = np.array(deltas_data).T
    
    psis= np.array(psis).T
    deltas= np.array(deltas).T

    Errors_psi = np.abs((psis - psis_data))
    Errors_delta = np.abs((deltas - deltas_data))
    Error_psi = Errors_psi.flatten()/(np.sqrt(len(psis_data.flatten()))) * 180 / np.pi
    Error_delta = Errors_delta.flatten()/(np.sqrt(len(deltas_data.flatten()))) * 180 / np.pi
    MSE_SE = 0.5*np.sum((alpha_psi*Error_psi)**2) + 0.5*np.sum((alpha_delta*Error_delta)**2)
    # print('current params', params)
    return  MSE_RT, MSE_SE



"""
Upload of nk data from ellipsometry fit, ellipsometry measurement and substrate nk data

"""   
def load(theta_values, minimal, maximal):
    lam_vacsubs_RT_values, M_RT_real, M_RT_imag = ld.load_nksubsdata()
    lam_vac_ellips, n_ellips, k_ellips = ld.load_nkellips()    
    lam_vac_RT, R_data, T_data = ld.load_data(minimal, maximal)   
    psis_data = []
    deltas_data = []
    for theta in theta_values:
        lam_vac_psidelta, psi_data,delta_data = ld.load_psidelta(minimal, maximal, theta)
        psis_data.append(psi_data)
        deltas_data.append(delta_data)
    return  (lam_vacsubs_RT_values,
            M_RT_real,
            M_RT_imag,
            lam_vac_ellips,
            n_ellips,
            k_ellips,
            lam_vac_RT,
            R_data, T_data,
            lam_vac_psidelta,
            psis_data,
            deltas_data)
#example cauchy
def plot_nk(lam_examp, params, lam_vac_ellips, n_opt_values, k_opt_values, n_ellips, k_ellips, comment=None):
    # Eg, C, E0, A, eps_1_inf, d_f, d_r1 = params
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(lam_examp, n_opt_values, label='$n$', color='orange', linewidth=0.7)
    ax.plot(lam_vac_ellips, n_ellips, color='orange', linestyle='--', linewidth=0.7)
    ax.plot(lam_examp, k_opt_values, label='$k$', color='red', linewidth=0.7)
    ax.plot(lam_vac_ellips, k_ellips, color='red', linestyle='--', linewidth=0.7)
    ax.set_ylabel('n/k', fontsize=14)
    ax.set_xlabel('$\lambda [nm]$', fontsize=14)
    ax.set_title('Tauc-Lorentz simulation of nk', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(0.8, 0.98), fontsize=14)
    
    # param_text = '\n'.join([f'{param}: {round(value, 2)}' for param, value in zip(['$E_g$', '$C$', '$E_0$', '$A$', r'$\varepsilon_1(\infty)$', '$d_f$', '$d_r1$'], params)])
    # ax.text(0.97, 0.5, param_text, transform=ax.transAxes, fontsize=12, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.7))

    if comment:
        fig.text(0.5, 0.01, f'{comment}', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    return fig
def plot_RT(lam_examp, ER_RT, T_tmm_opt, lam_vac_values, T_data, R_tmm_opt, R_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    MSE = ER_RT
    ax.text(0.98, 0.98, f'MSE={MSE:.2f}', transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.plot(lam_examp, T_tmm_opt*100, label='$T$', color='black', linewidth=0.7)
    ax.plot(lam_vac_values, T_data*100, color='black', marker='o', markersize=0.8, linestyle='none', markevery=2)
    ax.plot(lam_examp, R_tmm_opt*100, label='$R$', color='blue', linewidth=0.7)
    ax.plot(lam_vac_values, R_data*100,  color='blue', marker='o', markersize=0.8, linestyle='none', markevery=2)
    ax.set_title('Transfer-Matrix-Method RT', fontsize=14)
    ax.set_ylabel('R/T', fontsize=14)
    ax.set_xlabel('$\lambda [nm]$', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=14)
    return fig


def plot_psi_delta(lam_examp, lam_vac_psidelta, theta_values, psi_opt, delta_opt, psis_data, deltas_data, ER1, ER2):
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    ER_psi = ER1
    ER_delta = ER2
    colors = ['black', 'green', 'purple']
    
    for i, theta in enumerate(theta_values):
        ax1.plot(lam_examp, psi_opt[:, i]*180/np.pi, label=f'${theta}°$', color=colors[i], linewidth=0.7)
        ax1.plot(lam_vac_psidelta, psis_data[i]*180/np.pi, color=colors[i], marker='o', markersize=0.8, linestyle='none', markevery=3)
    
        ax2.plot(lam_examp, delta_opt[:, i]*180/np.pi, label=f'${theta}°$', color=colors[i], linewidth=0.7)
        ax2.plot(lam_vac_psidelta, deltas_data[i]*180/np.pi, color=colors[i], marker='o', markersize=0.8, linestyle='none', markevery=3)

    ax1.text(0.98, 0.98, f'MSE={ER_psi:.2f}', transform=ax1.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax1.set_ylabel('$\psi$', fontsize=14)
    ax1.set_xlabel('$\lambda [nm]$', fontsize=14)

    ax1.set_title('Transfer-Matrix-Method $\psi$', fontsize=14)
    ax1.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02), fontsize=14)
    
    ax2.text(0.98, 0.98, f'MSE={ER_delta:.2f}', transform=ax2.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax2.set_xlabel('$\lambda [nm]$', fontsize=14)
    ax2.set_ylabel('$\Delta$', fontsize=14)
    ax2.set_title('Transfer-Matrix-Method $\Delta$', fontsize=14)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=14)
    
 
    return fig1, fig2

#example cauchy
# def main():

#     DiE_model = 'Cauchy'
#     input_data = {}
#     n_3 = 1
#     k_3 = 0

#     c_list = ['i','c','i','i']
#     theta_0_R = 8
#     theta_0_T = 8
#     d_s_RT = 1.1e6
    
#     alpha_R = 0.5
#     alpha_T = 0.5
#     alpha_psi = 0.5
#     alpha_delta = 0.5
#     minimal, maximal = ld.lambda_range()
#     theta_values = [50]
#     (lam_vacsubs_RT_values,
#         M_RT_real,
#         M_RT_imag,
#         lam_vacsubs_ell_values,
#         M_ell_real, 
#         M_ell_imag, 
#         lam_vac_ellips,
#         n_ellips,
#         k_ellips,
#         lam_vac_RT,
#         R_data, T_data,
#         lam_vac_psidelta,
#         psis_data,
#         deltas_data) = load(theta_values, minimal, maximal)
#     a, b, d_f = [2.665100993291354, -41169.685857665296, 206.48727460792819]
#     input_data = {}
#     input_data['params'] = a, b
#     input_data['args'] = ({'d_s_RT': d_s_RT, 
#                                 'theta_0_T': theta_0_T, 
#                                 'theta_0_R': theta_0_R, 
#                                 'c_list': c_list, 
#                                 'n_3': n_3, 
#                                 'k_3': k_3, 
#                                 'lam_vac_RT': lam_vac_RT, 
#                                 'R_data': R_data, 
#                                 'T_data': T_data, 
#                                 'lam_vacsubs_RT_values': lam_vacsubs_RT_values, 
#                                 'M_RT_real': M_RT_real, 
#                                 'M_RT_imag': M_RT_imag, 
#                                 'lam_vacsubs_ell_values': lam_vacsubs_ell_values, 
#                                 'M_ell_real': M_ell_real, 
#                                 'M_ell_imag': M_ell_imag,
#                                 'theta_values': theta_values, 
#                                 'lam_vac_psidelta': lam_vac_psidelta, 
#                                 'psis_data': psis_data, 
#                                 'deltas_data': deltas_data, 
#                                 'alpha_R': alpha_R, 
#                                 'alpha_T': alpha_T, 
#                                 'alpha_psi': alpha_psi, 
#                                 'alpha_delta': alpha_delta, 
#                                 'DiE_model': DiE_model})
     

#     args = input_data['args']
#     params = a, b, d_f
#     print('Parameters', list(input_data['params']))
#     params_cauchy = a, b
#     print("n(632 nm)", cauchy(632, *params_cauchy)[2])
    
#     num_theta = len(theta_values)
#     lam_examp = np.linspace(minimal, maximal, 100)
    
#     eps_1_opt_values, eps_2_opt_values, n_opt_values, k_opt_values = np.transpose(np.array([cauchy(lam_vac, *params_cauchy) for lam_vac in lam_examp]))
#     psi_opt = np.zeros((len(lam_examp), num_theta))
#     delta_opt = np.zeros((len(lam_examp), num_theta))
        
#         # Schleife über Theta-Werte
#     for i, theta in enumerate(theta_values):
#         psi_opt[:, i], delta_opt[:, i] = np.transpose(np.array([model_2(lam_vac, params, args, theta) for lam_vac in lam_examp]))
    


    
#     # plot_nk(lam_examp, params, lam_vac_ellips, n_opt_values, k_opt_values, n_ellips, k_ellips)
#     plot_psi_delta(lam_examp, lam_vac_psidelta, theta_values, psi_opt, delta_opt, psis_data, deltas_data, 1, 1)

#example TL   
def main():
    
    """
    In the main function the optimization is done using the minimize function for minimizing the error.
    
    The results are calculated using the optimized Tauc-Lorentz parameters and the film thickness.
    The results are shown in graphics.
    """
    input_data = {}
    n_osz = 2

    #Materialkonstanten
    d_s_RT = 1.1e6
    # d_s_el = 1.1e6
    c_list = ['i','c', 'c', 'i','i']
    theta_0_R = 8
    theta_0_T = 0

    alpha_R = 1
    alpha_T = 1
    alpha_psi =1
    alpha_delta = 1


    theta_values = [50,60,70]
    minimal, maximal = ld.lambda_range()
    

    (lam_vacsubs_values, M_real, M_imag,
     lam_vac_ellips, n_ellips, k_ellips,
         lam_vac_RT, R_data, T_data,
           lam_vac_SE, psis_data, deltas_data) = load(theta_values, minimal, maximal)
    lam_examp = lam_vac_SE
    R_data_reg = interp1d(lam_vac_RT, R_data, kind='linear')(lam_examp) 
    T_data_reg = interp1d(lam_vac_RT, T_data, kind='linear')(lam_examp)
    
    n_s = interp1d(lam_vacsubs_values, M_real[:, 1], kind='linear')(lam_examp) 
    k_s = interp1d(lam_vacsubs_values, M_imag[:, 1], kind='linear')(lam_examp)
    
    Ns_RT = n_s + 1j* k_s
    
    n_s2 = interp1d(lam_vacsubs_values, M_real[:, 1], kind='linear')(lam_examp) 
    k_s2 = interp1d(lam_vacsubs_values, M_imag[:, 1], kind='linear')(lam_examp)
    

    Ns_SE = n_s2 + 1j* k_s2

    subs_RT = np.array([lam_examp, Ns_RT]).T
    subs_SE = np.array([lam_examp, Ns_SE]).T


    comment = 'SE50,60,70, Startwerte aus erstem fit (se50)'
    input_data = {}
    input_data['params'] =  [0.0092, 
                             0.3274, 0.3020,
                             1.3955, 15.6506, 11.0666, 69.4391, 
                             2.1301, 2.0829, 3.9080, 90.9656, 
                             0.6805, 200.1364, 11.1245]
    
    input_data['args'] = ({'d_s_RT': d_s_RT, 
                            'theta_0_T': theta_0_T, 
                            'theta_0_R': theta_0_R, 
                            'c_list': c_list, 
                            'lam_vac_RT': lam_vac_RT, 
                            'R_data': R_data, 
                            'T_data': T_data, 
                            'R_data_reg': R_data_reg, 
                            'T_data_reg': T_data_reg,
                            'subs_RT': subs_RT, 
                            'subs_SE':subs_SE,
                            'theta_values': theta_values, 
                            'lam_vac_examp': lam_examp,
                            'lam_vac_psidelta': lam_vac_SE, 
                            'psis_data': psis_data, 
                            'deltas_data': deltas_data, 
                            'alpha_R': alpha_R, 
                            'alpha_T': alpha_T, 
                            'alpha_psi': alpha_psi, 
                            'alpha_delta': alpha_delta,
                            'n': n_osz
                            })
 
    params = input_data['params']
    args = input_data['args']
    #results 
    MSE_R, MSE_T = error_RT(**input_data)
    MSE_RT = MSE_R + MSE_T
    MSE_psi , MSE_delta = error_ellips(**input_data)
    MSE_SE = MSE_psi + MSE_delta

    params_x= np.zeros(len(params)+1)
    params_x[0] = n_osz
    params_x[1:-3] = params[0:-3]
    
    print("MSE RT/SE", (MSE_RT, MSE_SE))
    print('Parameters', '\n'.join(map(str, input_data['params'])))
    print("n(632 nm)", TL_drude(632, params_x)[2])

    eps_1_opt_values, eps_2_opt_values, n_opt_values, k_opt_values = np.transpose(np.array([TL_drude(lam_vac, params_x) for lam_vac in lam_examp]))


    
    R_tmm_opt = []
    T_tmm_opt = []
    
    R_tmm_opt, T_tmm_opt= (np.array([model(lam_vac, params, args) for lam_vac in lam_examp])).T
    # Anzahl der Theta-Werte
    num_theta = len(theta_values)
    
    # Arrays für Psi- und Delta-Optimalwerte
    psi_opt = np.zeros((len(lam_examp), num_theta))
    delta_opt = np.zeros((len(lam_examp), num_theta))
    
    # Schleife über Theta-Werte
    for i, theta in enumerate(theta_values):
        psi_opt[:, i], delta_opt[:, i] = np.transpose(np.array([model_2(lam_vac, params, args, theta) for lam_vac in lam_examp]))
    
    h = 6.626e-34  # planksches Wirkungsquantum
    c0 = 2.988e8  # Lichtgeschwindigkeit im Vakuum
    q = 1.602e-19 #Elemntarladung
         
    E = h * c0 / (lam_examp* 1e-9)/q
    
    results = (lam_examp, n_opt_values, k_opt_values)
    
    np.savetxt('nk_RT', results)

    plot_nk(lam_examp, params, lam_vac_ellips, n_opt_values, k_opt_values, n_ellips, k_ellips, comment=comment)
    plot_RT(lam_examp, MSE_RT, T_tmm_opt, lam_vac_RT, T_data, R_tmm_opt, R_data)
    plot_psi_delta(lam_examp, lam_vac_SE, theta_values, psi_opt, delta_opt, psis_data, deltas_data, MSE_psi, MSE_delta)    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
       main()
       

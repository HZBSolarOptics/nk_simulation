from scipy.optimize import  minimize, basinhopping, least_squares

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import load_data as ld
from IPython.display import clear_output

from dielectric_models import cauchy, TL, TL_drude, TL_multi
import os

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

#Materialkonstanten

def error_TL(params, *args):
    n_osz, lam_vac_nk, n_data, k_data, alpha_n, alpha_k = args

    
    
    _, _, n_modell_values,k_modell_values = np.transpose(np.array([TL(lam_vac, params) for lam_vac in lam_vac_nk]))
    Error_n = (n_modell_values - n_data)**2
    Error_k = (k_modell_values - k_data)**2
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    
    # Drucke die aktuellen Parameter
    print('Parameters:', (', '.join([format(param, '.2f') for param in params])))
    # Tauc-Lorentz Plot
    axs.plot(lam_vac_nk,n_modell_values, label='$n_{calc}$', color='orange', linewidth=0.7)
    axs.plot(lam_vac_nk, n_data, label='$n_{data}$', color='black', linestyle ='--')
    axs.plot(lam_vac_nk, k_modell_values, label='$k_{calc}$', color='red', linewidth=0.7)
    axs.plot(lam_vac_nk, k_data, label='$k_{data}$', color='black', linestyle ='--')
    axs.set_xlabel('$\lambda [nm]$')
    axs.set_ylabel('nk')
    axs.set_title('Tauc Lorentz fit to n(cauchy) values')
    axs.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    
    return np.concatenate((Error_n, Error_k))

def error_cauchy(params, *args):
    _, lam_vac_nk, n_data, k_data, alpha_n, alpha_k = args

    _, _, n_modell_values,k_modell_values = np.transpose(np.array([cauchy(lam_vac, params) for lam_vac in lam_vac_nk]))
    Error_n = (n_modell_values - n_data)**2

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    
    # Drucke die aktuellen Parameter
    print('Parameters:', (', '.join([format(param, '.2f') for param in params])))
    # Tauc-Lorentz Plot
    axs.plot(lam_vac_nk,n_modell_values, label='$n_{calc}$', color='orange', linewidth=0.7)
    axs.plot(lam_vac_nk, n_data, label='$n_{data}$', color='black', linestyle ='--')
    axs.set_xlabel('$\lambda [nm]$')
    axs.set_ylabel('n')
    axs.set_title('cauchy fit')
    axs.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    return Error_n


def main():
    
    alpha_n = 0.5
    alpha_k = 0.5
    n_osz = 1
    # Initial guess for parameters
   # lam_vac_nk = np.linspace(800, 2000)
    #_, _, n_data, k_data = np.transpose(np.array([cauchy(lam_vac, ( 2.460184, 127669.21946)) for lam_vac in lam_vac_nk]))
   
    
    lam_vac_nk, n_data, k_data = ld.load_nk('Glass')

    args = (n_osz, lam_vac_nk, 
            n_data, k_data, 
            alpha_n, alpha_k)
    
    def callback_function(params, f, accepted):
        # Diese Funktion wird nach jeder Iteration aufgerufen
        print(f"Iteration: {callback_function.iteration}, Params: {params}, Objective Value: {f}, Accepted: {accepted}")
        callback_function.iteration += 1
    
    callback_function.iteration = 0
    #initial_guess = np.array([1,2,1,1,50])
    #opt_param = least_squares(error_TL, initial_guess, args=args, method = 'lm')
    initial_guess = np.array([1.50, 4482.00])
    opt_param = least_squares(error_cauchy, initial_guess, args=args, method = 'lm')
    # a_opt, b_opt = opt_param.x
    print(opt_param)
    print("Optimal Parameter", list(opt_param.x))


    # # Display the results using the optimized parameters
    lam_examp = np.linspace(100,2000,100)
    
   
    _, _, n_opt_values, k_opt_values = np.transpose(np.array([cauchy(lam_vac, opt_param.x) for lam_vac in lam_examp]))
    
    nk_data = []
    nk_data.append(lam_examp)
    nk_data.append(n_opt_values)
    nk_data.append(k_opt_values)
    np.savetxt('interpolated_nkdata.csv', np.array(nk_data).T, delimiter=';')


    
    # Einstellungen für die Layout-Anpassung
    plt.tight_layout()
    
    # Zeige die Plots an
    plt.show()    


if __name__ == "__main__":
    main()  

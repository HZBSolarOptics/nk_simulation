import numpy as np
import matplotlib.pyplot as plt
import math
def TL_drude(lam_vac, params):
    
    n_osz = int(params[0])
    eps_1_inf = params[1]
    A = params[2]
    L = params[3]
    eps_TL = 0
    
    if any(param < 0 for param in params):
        eps_total_1 = 10e10
        eps_total_2 = 10e10
        n = 10e10
        k = 0
    else:    
        for i in range(n_osz):
            # Indexberechnung für die Parameter des aktuellen Oszillators
            
            start_idx = 4 + i * 4
            end_idx = start_idx + 4
            osc_params = params[start_idx:end_idx]
            # Aufruf der epsilon-Funktionen mit den aktuellen Oszillatorparametern
            eps_1_TL = epsilon_1(lam_vac, osc_params)
            eps_2_TL = epsilon_2(lam_vac, osc_params)
            
            # Aufaddieren der Ergebnisse zu den Gesamtsummen
            eps_TL_uno = eps_1_TL + 1j * eps_2_TL
            
            eps_TL += eps_TL_uno
        eps_drude = drude(lam_vac, (A,L))[0] + 1j *  drude(lam_vac, (A,L))[1] 
        eps_total = eps_1_inf + eps_TL + eps_drude
       
        eps_total_1 = eps_total.real
        eps_total_2 = eps_total.imag
            
        n, k = nk(lam_vac, eps_total_1, eps_total_2)
    
    return eps_total_1, eps_total_2, n, k
def TL_multi(lam_vac, params):
    if any(param < 0 for param in params):
        eps_total_1 = 10e10
        eps_total_2=10e10
        n = 10e10
        k=0
    else:
        n_osz = int(params[0])
        eps_1_inf = params[1]
        eps_TL = 0
        for i in range(n_osz):
            # Indexberechnung für die Parameter des aktuellen Oszillators
            
            start_idx = 2 + i * 4
            end_idx = start_idx + 4
            osc_params = params[start_idx:end_idx]
           
            # Aufruf der epsilon-Funktionen mit den aktuellen Oszillatorparametern
            eps_1_TL = epsilon_1(lam_vac, osc_params)
            eps_2_TL = epsilon_2(lam_vac, osc_params)
            
            # Aufaddieren der Ergebnisse zu den Gesamtsummen
            eps_TL_uno = eps_1_TL + 1j * eps_2_TL
            
            eps_TL += eps_TL_uno
        
        eps_total = eps_1_inf + eps_TL

        eps_total_1 = eps_total.real
        eps_total_2 = eps_total.imag
            
        n, k = nk(lam_vac, eps_total_1, eps_total_2)
    if any(params) < 0:
        n = 10e10
    return eps_total_1, eps_total_2, n, k

def epsilon_2(lam_vac, params):
    h = 6.626e-34  # planksches Wirkungsquantum
    c0 = 2.988e8  # Lichtgeschwindigkeit im Vakuum
    q = 1.602e-19 #Elemntarladung
         
    E = h * c0 / (lam_vac* 1e-9)/q
    Eg, C, E0, A = params  
    
    result = A * E0 * C * (E - Eg)**2 / ((E**2 - E0**2)**2 + C**2 * E**2) * (1 / E)
    eps_2 = np.where(E > Eg, result, 0)
    return eps_2   
    
def epsilon_1(lam_vac, params):
    h = 6.626e-34  # planksches Wirkungsquantum
    c0 = 2.988e8  # Lichtgeschwindigkeit im Vakuum
    q = 1.602e-19 #Elemntarladung
         
    E = h * c0 / (lam_vac* 1e-9)/q
    Eg, C, E0, A = params
    a1 = (Eg**2 - E0**2) * E**2
    a2 = Eg**2 * C**2
    a3 = -E0**2 * (E0**2 + 3 * Eg**2)
    a_1= a1 + a2 + a3
          
    b1 = (E**2 - E0**2) * (E0**2 + Eg**2)
    b2 = Eg**2 * C**2
    a_2 = b1 + b2
    
    alpha = np.sqrt(4 * E0**2 - C**2)
    
    gamma = np.sqrt(E0**2 - C**2 / 2)
    
    c1 = np.power(np.power(E, 2) - gamma**2, 2)
    c2 = 0.25 * alpha**2 * C**2
    z= np.power(c1 + c2, 0.25)
    t2 = A * C * a_1 / (2 * np.pi * z**4 * alpha * E0) * np.log((E0**2 + Eg**2 + alpha * Eg) / (E0**2 + Eg**2 - alpha * Eg))
    t3 = -A * a_2 / (np.pi * z**4 * E0) * (np.pi - np.arctan(1 / C * (2 * Eg + alpha)) + np.arctan(1 / C * (alpha - 2 * Eg)))
    t4 = 4 * A * E0 * Eg * (E**2 - gamma**2) / (np.pi * z**4 * alpha) * (np.arctan(1 / C * (alpha + 2 * Eg)) + np.arctan(1 / C * (alpha - 2 * Eg)))
    t5 = -A * E0 * C * (E**2 + Eg**2) / (np.pi * z**4 * E) * np.log(np.fabs(E - Eg) / (E + Eg))
    t6 = 2 * A * E0 * C * Eg / (np.pi * z**4) * np.log(np.fabs(E - Eg) * (E + Eg) / ((E0**2 - Eg**2)**2 + Eg**2 * C**2)**0.5)
    return t2 + t3 + t4 + t5 + t6

def nk(lam_vac, eps_1, eps_2):
    n = np.sqrt((eps_1 + (eps_1**2 + eps_2**2)**(0.5)) / 2)
    k = np.sqrt((-eps_1 + (eps_1**2 + eps_2**2)**(0.5)) / 2)      
    return n, k

def TL(lam_vac, params):
    
    if any(param < 0 for param in params):
        eps_1 = 10e10
        eps_2 = 10e10
        n = 10e10
        k=0
    else:

        eps_1 = params[0] +  epsilon_1(lam_vac, params[1:])           
 
        eps_2 = epsilon_2(lam_vac, params[1:])
        n, k = nk(lam_vac, eps_1, eps_2)
        
    return eps_1, eps_2, n, k

def drude(lam_vac, params):
    A, L = params
    h = 6.626e-34  # planksches Wirkungsquantum
    c0 = 2.988e8  # Lichtgeschwindigkeit im Vakuum
    q = 1.602e-19 #Elemntarladung
         
    E = h * c0 / (lam_vac* 1e-9)/q

    eps = - A / (1j * E * L + E**2)
    eps1 = eps.real
    eps2 = eps.imag
    return eps1, eps2

def cauchy(lam_vac, params):
    if any(param < 0 for param in params):
        epsilon_1 = 10e10
        epsilon_2 = 0
        n = 10e10
        k=0
    else:
        n_osz = int(params[0])
        eps_inf = params[1]
        if n_osz == 1:

            n = params[2] + params[3] / (lam_vac**2) 
        
        elif n_osz == 2:
            
            n = params[2] + params[3] / (lam_vac**2) + params[4] / (lam_vac**4) 
        
        elif n_osz == 3:
            
            n = params[2] + params[3] / (lam_vac**2) + params[4] / (lam_vac**4) - params[5] * (lam_vac**2)
        k = 0
        epsilon_1 = n**2 + eps_inf
        epsilon_2 = 0
    return epsilon_1, epsilon_2, n, k

def sellmeier(lam_vac, *params):
    if any(param < 0 for param in params):
        epsilon_1 = 10e10
        epsilon_2 = 10e10
        n = 10e10
        k=0
    else:
        n = int(params[0])
        eps_inf = params[1]
        e_SM = 0
        for i in range(n):
            
            start = 2*i + 2
            end = start + 1
            
            SM = params[start] * lam_vac**2 / (lam_vac**2 - params[end]**2)
            
            e_SM += SM
            
        epsilon_1 = eps_inf + e_SM
                     
        epsilon_2 = 0
        
        n = epsilon_1**0.5
        k = 0
    return epsilon_1, epsilon_2, n, k

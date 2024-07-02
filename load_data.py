# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:47:54 2024

@author: a4246
"""
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import interp1d

def lambda_range():
    
    minimal = float(input("Please define the minimum wavlength (nm): "))
    maximal = float(input("Please define the maximum wavlength (nm): "))
    if maximal < minimal:
        print('The maximal value must be higher than the minimal value!')
    if maximal - minimal < 100:
        print('The wavelength range must be atleast 100 nm!')
    return minimal, maximal

def load_RT(minimal, maximal):
    
    root = tk.Tk()
    root.withdraw()  # Verstecke das Hauptfenster, da wir es nicht verwenden
    
    # Dateiauswahl mit dem Dateiauswahldialog von Tkinter
    file_path = filedialog.askopenfilename(title="Choose R/T-Measurement .csv file", 
                                           filetypes=[("CSV-Dateien", "*.csv")])
    
    if not file_path:
        print("No file selected.")
        return None, None, None, None, None, None, 'canceled'

    try:

        RT_data = np.genfromtxt(file_path, delimiter=';')
        
        lam_vac_raw = RT_data[:,0]
        T_raw = RT_data[:,1]
        R_raw = RT_data[:,2]
    
        # Suche nach den Zeilen im Wellenlängenbereich für measured_data
        measured_min_index = np.searchsorted(RT_data[:, 0], minimal, side='left')
        measured_max_index = np.searchsorted(RT_data[:, 0], maximal, side='right')
    
        # Extrahiere die relevanten Daten für measured_data
        lam_vac_RT = RT_data[measured_min_index:measured_max_index, 0]
        T_data = RT_data[measured_min_index:measured_max_index, 1]
        R_data = RT_data[measured_min_index:measured_max_index, 2]
        
        root.destroy()
    
        return lam_vac_raw, R_raw, T_raw, lam_vac_RT, R_data, T_data, file_path
    except Exception:
        root.destroy()
        return None, None, None, None, None, None, None
def load_SE(minimal, maximal):
    
    """
    Input of ellipsometry measurement as .csv file.
    The File needs to contain three columns, separatet with a ";"
    the columns do NOT contain headlines.
    Decimal separator: "."
    
    Every row should look like this 
    
    
    [wavelength (nm)];[psi (rad)];[Delta(rad)]
    
    example
    
    189.80969;29.44986;143.36569
    190.6175;29.20235;141.72265
    198.69563;29.27197;139.49503
    210.81281;29.57919
    
    """   
    root = tk.Tk()
    root.withdraw()  # Verstecke das Hauptfenster, da wir es nicht verwenden
    
    # Dateiauswahl mit dem Dateiauswahldialog von Tkinter
    file_path = filedialog.askopenfilename(title="Choose SE-measurement file ", filetypes=[("CSV-Dateien", "*.csv")])
    
    if not file_path:
        print("No file selected.")
        return None, None, None, None, None, None,None, 'canceled'

    try:

        header = np.genfromtxt(file_path, delimiter=';', max_rows=1)
        theta_values = header
        
        # Laden der gemessenen Werte aus der CSV-Datei für measured_data
        SE_data = np.genfromtxt(file_path, skip_header = 1, delimiter=';')
    
        # Suche nach den Zeilen im Wellenlängenbereich für measured_data
        SE_min_index = np.searchsorted(SE_data[:, 0], minimal, side='left')
        SE_max_index = np.searchsorted(SE_data[:, 0], maximal, side='right')
    
        # Extrahiere die relevanten Daten für measured_data
        lam_vac_raw = SE_data[:, 0]
        lam_vac_SE = SE_data[SE_min_index:SE_max_index, 0]
        
        psis_data = []
        deltas_data = []
        
        psis_raw = []
        deltas_raw = []
        
        for i in range(len(theta_values)): 
            index_psi = 2*i+1
            index_delta = index_psi + 1
            
            delta = SE_data[SE_min_index:SE_max_index, index_delta]
            psi = SE_data[SE_min_index:SE_max_index, index_psi]
            
            delta_raw = SE_data[:, index_delta]
            psi_raw = SE_data[:, index_psi]
            
            psis_data.append(psi)
            deltas_data.append(delta)
            
            psis_raw.append(psi_raw)
            deltas_raw.append(delta_raw)
            
        root.destroy()
        return theta_values, lam_vac_raw, psis_raw, deltas_raw, lam_vac_SE, psis_data, deltas_data, file_path
    except Exception:
        root.destroy()
        return None, None, None, None, None, None,None, None

def load_nk(sample):
    
    """
    Input of ellipsometry nk fit as .csv file.
    The File needs to contain three columns, separatet with a ";"
    the columns do NOT contain headlines.
    Decimal separator: "."
    
    Every row should look like this    
    
    [wavelength (nm)];[n];[k]
    
    """   
    file_path_nke = r'CSV for python\Referenz\{}.csv'.format(sample)


    # Load measured values from the CSV file
    nk_data = np.genfromtxt(file_path_nke, delimiter=';')

    # Extrahiere die relevanten Daten für nks_data
    lam_vac_ellips = nk_data[:, 0]
    n_ellips_data = nk_data[:, 1]
    k_ellips_data = nk_data[:, 2]
    
    return lam_vac_ellips, n_ellips_data, k_ellips_data


def load_nksubsdata ():
    
    """
    Input of substrate nk data as .csv file.
    The File needs to contain three columns, separatet with a ";"
    the columns do NOT contain headlines.
    Decimal separator: "."
    
    Every row should look like this 
    
    
    [wavelength (nm)];[n];[k]

    
    """   
   
    
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Prompt the user to select a file
    root.wm_attributes('-topmost', 1)
    file_path = filedialog.askopenfilename(parent=root,
        title="Select your substrate nk-Data .csv File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],)
    root.wm_attributes('-topmost', 0)
    if not file_path:
        print("No file selected.")
        return None, None, None, 'canceled'

    # Load measured values from the CSV file
    try:
        nks_data = np.genfromtxt(file_path, delimiter=';')
        lam_vacsubs_values = nks_data[:, 0] 
        ns_data = nks_data[:, 1] 
        ks_data = nks_data[:, 2] 
        Ns_data = np.array([complex(ns,ks) for ns, ks in zip(ns_data, ks_data)])
    
        M_real = np.column_stack((lam_vacsubs_values, np.real(Ns_data)))
        M_imag = np.column_stack((lam_vacsubs_values, np.imag(Ns_data)))
    
        root.destroy()  # Close the Tkinter window
        return lam_vacsubs_values, M_real, M_imag, file_path
    except Exception:
        root.destroy()
        return None, None, None, None

def load_subs(sample):
    
    """
    Input of substrate nk data as .csv file.
    The File needs to contain three columns, separatet with a ";"
    the columns do NOT contain headlines.
    Decimal separator: "."
    
    Every row should look like this 
    
    
    [wavelength (nm)];[n];[k]

    
    """   
   
    
    file_path = r'CSV for python\Substrate\{}.csv'.format(sample)

    # Load measured values from the CSV file
    nks_data = np.genfromtxt(file_path, delimiter=';')
    lam_vacsubs_values = nks_data[:, 0] 
    ns_data = nks_data[:, 1]
    ks_data = nks_data[:, 2] 

    Ns_data = np.array([complex(ns,ks) for ns, ks in zip(ns_data, ks_data)])

    M_real = np.column_stack((lam_vacsubs_values, np.real(Ns_data)))
    M_imag = np.column_stack((lam_vacsubs_values, np.imag(Ns_data)))

    return lam_vacsubs_values, M_real, M_imag
def reg_RT(lam_vac_reg, *input_data):
    lam_vac_RT, R_data, T_data = input_data
    R = interp1d(lam_vac_RT, R_data, kind='linear', fill_value='extrapolate')(lam_vac_reg)
    T = interp1d(lam_vac_RT, T_data, kind='linear', fill_value='extrapolate')(lam_vac_reg)
    return R, T

def reg_SE(lam_vac_reg, *input_data):
    lam_vac_SE, psis_data, deltas_data = input_data
    psi = interp1d(lam_vac_SE, psis_data, kind='linear', axis=0, fill_value='extrapolate')(lam_vac_reg)
    delta = interp1d(lam_vac_SE, deltas_data, kind='linear', axis=0, fill_value='extrapolate')(lam_vac_reg)
    return psi, delta



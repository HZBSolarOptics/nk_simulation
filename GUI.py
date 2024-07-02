import sys
import os
import pyqtgraph as pg
import csv
import tkinter as tk
from tkinter import filedialog


import numpy as np
import load_data as ld
from scipy.interpolate import interp1d

from scipy.optimize import least_squares

from PyQt5 import uic, QtWidgets,  QtGui, QtCore
from PyQt5.QtCore import QThread, pyqtSignal, QObject

from Calculation import model, model_2
from fit_TLD import RT, RT_rough, SE, SE_rough
from fit_TL import RT_TL, RT_rough_TL, SE_TL, SE_rough_TL
from fit_cauchy import RT_C, RT_rough_C, SE_C, SE_rough_C
from fit_sellmeier import RT_SM, RT_rough_SM, SE_SM, SE_rough_SM


from dielectric_models import TL_multi, TL_drude, cauchy, sellmeier



current_path = os.path.dirname(os.path.realpath(__file__))
GUI_path  = os.path.join(current_path, 'pyqt5')
os.chdir(GUI_path)
Ui_MainWindow_settings, QTBaseClass = uic.loadUiType('settings.ui')
Ui_MainWindow_results, _ = uic.loadUiType('results.ui')
os.chdir(current_path)

class Settings(QtWidgets.QMainWindow, Ui_MainWindow_settings):
    def __init__(self):
        super(Settings, self).__init__()
        self.setupUi(self)

        self.model_inputs = {} 

        self.RT_inputs = {}
        self.SE_inputs = {}

        self.results = {}
        self.setup_connections()
        
        self.thread = None
        self.worker = None

    def setup_connections(self):
        self.Bruggeman.clicked.connect(self.bruggeman)
        self.bruggeman_enabled = False
        #Upload checks invisible
        
        self.RT_upload_check.setVisible(False)
        self.RT_upload_nk_check.setVisible(False)
        self.SE_upload_nk_check.setVisible(False)
        self.SE_upload_check.setVisible(False)
        
    	#upload 
        self.error_found = False

        self.RT_upload.clicked.connect(self.load_RT)
        self.SE_upload.clicked.connect(self.load_SE)
        
        self.RT_min.textChanged.connect(self.change_RT_range)
        self.RT_max.textChanged.connect(self.change_RT_range)
        self.SE_min.textChanged.connect(self.change_SE_range)
        self.SE_max.textChanged.connect(self.change_SE_range)


        #upload substrate
        self.RT_upload_nk.clicked.connect(self.load_subs_RT)
        self.SE_upload_nk.clicked.connect(self.load_subs_SE)

        #delete
        self.RT_delete.clicked.connect(self.delete_RT)
        self.RT_delete_subs.clicked.connect(self.delete_subs_RT)
        self.SE_delete.clicked.connect(self.delete_SE)
        self.SE_delete_subs.clicked.connect(self.delete_subs_SE)

        #Help
        self.help_RT.clicked.connect(self.HW_RT)
        self.help_SE.clicked.connect(self.HW_SE)
        self.help_RT_tmm.clicked.connect(self.HW_RT_tmm)
        self.help_SE_tmm.clicked.connect(self.HW_SE_tmm)

        self.help_subs_RT.clicked.connect(self.HW_subs)
        self.help_subs_SE.clicked.connect(self.HW_subs)
        self.help_angle.clicked.connect(self.HW_angle)
        self.help_model.clicked.connect(self.HW_model)
        self.help_TL.clicked.connect(self.HW_TL)
        self.help_cauchy.clicked.connect(self.HW_cauchy)
        self.help_SM.clicked.connect(self.HW_SM)
        self.help_error.clicked.connect(self.HW_error)
        self.help_tol.clicked.connect(self.HW_tol)
        self.help_tresh.clicked.connect(self.HW_tresh)

        self.open_HB.clicked.connect(self.Helpbook)
        self.open_console.clicked.connect(self.reopen_console)

        self.main_button_startcalc.clicked.connect(self.reopen_console)

        self.main_button_startcalc.clicked.connect(self.inputs)
        self.main_button_startcalc.clicked.connect(self.inputs_RT)
        self.main_button_startcalc.clicked.connect(self.inputs_SE)
        self.main_button_startcalc.clicked.connect(self.Errors)

        self.show_results = None
    
        self.count = 0
        quit = QtWidgets.QAction("Quit", self)
        quit.triggered.connect(self.closeEvent)
   
    def close_result(self):
        if self.show_results is not None:
            self.show_results.close()
    def reopen_console(self):
        console_window.show()
    def closeEvent(self, event):
        close = QtWidgets.QMessageBox()
        close.setText("You sure you want to quit?")
        close.setWindowTitle('Exit')

        close.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        close = close.exec()
        
        if close == QtWidgets.QMessageBox.Yes:
            
            console_window.close()

            if self.thread is not None:
                self.thread.quit()
                self.thread.wait()
                self.thread = None

            if self.show_results is not None:
                self.show_results.close()
            
            event.accept()
        else:
            event.ignore()

    def open_results(self,model_inputs, SE_inputs, RT_inputs, results):
        if self.results_window is None:
            self.results_window = Result(model_inputs, SE_inputs, RT_inputs, results)
            self.results_window.show()
    def bruggeman(self):
        self.bruggeman_enabled = not self.bruggeman_enabled
        if self.bruggeman_enabled:
            self.thickness.setEnabled(True)
            self.airfraction.setEnabled(True)
            self.d_r.setEnabled(True)
            self.f_1.setEnabled(True)
        else:
            self.thickness.setEnabled(False)
            self.airfraction.setEnabled(False)
            self.d_r.setEnabled(False)
            self.f_1.setEnabled(False)
    
    def load_subs_RT(self):
        print('load RT substrate')

        try:
            lam_vacsubs_values, M_real, M_imag, file_path = ld.load_nksubsdata()
            if file_path == 'canceled':
                return
            if lam_vacsubs_values is None or M_real is None or M_imag is None or file_path is None:
                self.error_found = 'True'
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load the data. Please check the file format and try again.")
                return
            self.RT_upload_nk_check.setVisible(True)
            self.RT_inputs['substrate lam_vac'] = lam_vacsubs_values
            self.RT_inputs['M_real'] = M_real
            self.RT_inputs['M_imag'] = M_imag     
            self.RT_subs_file.setText(file_path)    
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"An error occurred: {e}")

    def load_subs_SE(self):
        print('load SE substrate')

        try:
            lam_vacsubs_values, M_real, M_imag, file_path = ld.load_nksubsdata()
            if file_path == 'canceled':
                return
            if lam_vacsubs_values is None or M_real is None or M_imag is None or file_path is None:
                self.error_found = 'True'
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load the data. Please check the file format and try again.")
                return
            self.SE_upload_nk_check.setVisible(True)
    
            self.SE_inputs['substrate lam_vac'] = lam_vacsubs_values
            self.SE_inputs['M_real'] = M_real
            self.SE_inputs['M_imag'] = M_imag        
            self.SE_subs_file.setText(file_path)        
        
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"An error occurred: {e}")

    def load_RT(self):
       
        minimal = float(self.RT_min.text())
        maximal = float(self.RT_max.text())
        try:
            lam_vac_raw, R_raw, T_raw , lam_vac_RT, R_data, T_data, file_path = ld.load_RT(minimal, maximal)
            if file_path == 'canceled':
                return
            if lam_vac_raw is None or R_raw is None or T_raw is None or lam_vac_RT is None or R_data is None or T_data is None or file_path is None:
                self.error_found = 'True'
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load the data. Please check the file format and try again.")
                return
        
            print(f'load {file_path}')

            self.RT_upload_check.setVisible(True)
            self.RT_inputs['lam_vac'] = lam_vac_RT
            self.RT_inputs['R'] = R_data
            self.RT_inputs['T'] = T_data  
            
            self.RT_inputs['lam_vac_raw'] = lam_vac_raw
            self.RT_inputs['R_raw'] = R_raw 
            self.RT_inputs['T_raw'] = T_raw
            console_window.update_RT_plot_signal.emit(lam_vac_raw.tolist(),R_raw.tolist(),T_raw.tolist())
            self.RT_file.setText(file_path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"An error occurred: {e}")

    def load_SE(self):
        # Überprüfen, ob lambda_min und lambda_max festgelegt wurden
        print('load SE data')

        minimal = float(self.SE_min.text())
        maximal = float(self.SE_max.text())
        try:

            theta_values,  lam_vac_raw, psis_raw, deltas_raw, lam_vac_SE,psis_data, deltas_data, file_path  = ld.load_SE(minimal, maximal)
            if file_path == 'canceled':
                return
            if lam_vac_raw is None or psis_raw is None or deltas_raw is None or lam_vac_SE is None or psis_data is None or deltas_data is None or file_path is None:
                self.error_found = 'True'
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load the data. Please check the file format and try again.")
                return
            self.SE_upload_check.setVisible(True)
            
            self.SE_inputs['thetas'] = theta_values 
    
            self.SE_inputs['lam_vac'] = lam_vac_SE
            self.SE_inputs['deltas'] = deltas_data 
            self.SE_inputs['psis'] = psis_data
            
            self.SE_inputs['lam_vac_raw'] = lam_vac_raw
            self.SE_inputs['deltas_raw'] = deltas_raw 
            self.SE_inputs['psis_raw'] = psis_raw
            self.SE_file.setText(file_path)
            console_window.update_SE_plot_signal.emit(theta_values.tolist(),lam_vac_raw.tolist(),psis_raw,deltas_raw)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"An error occurred: {e}")
    def change_SE_range(self):
        
        if 'lam_vac' in self.SE_inputs:
            minimal = float(self.SE_min.text())
            maximal = float(self.SE_max.text())
            
            SE_min_index = np.searchsorted(self.SE_inputs['lam_vac_raw'], minimal, side='left')
            SE_max_index = np.searchsorted(self.SE_inputs['lam_vac_raw'], maximal, side='right')
    
            # Extrahiere die relevanten Daten für measured_data
            lam_vac_SE = self.SE_inputs['lam_vac_raw'][SE_min_index:SE_max_index]
            
            psis_data = []
            deltas_data = []
            for i in range(len(self.SE_inputs['thetas'])): 
                
                delta = self.SE_inputs['deltas_raw'][i][SE_min_index:SE_max_index]
                psi = self.SE_inputs['psis_raw'][i][SE_min_index:SE_max_index]
                
                psis_data.append(psi)
                deltas_data.append(delta)
            

            self.SE_inputs['lam_vac'] = lam_vac_SE
            self.SE_inputs['deltas'] = deltas_data 
            self.SE_inputs['psis'] = psis_data
        else:
            pass
    def change_RT_range(self):
        
        if 'lam_vac' in self.RT_inputs:
            minimal = float(self.RT_min.text())
            maximal = float(self.RT_max.text())
            
            RT_min_index = np.searchsorted(self.RT_inputs['lam_vac_raw'], minimal, side='left')
            RT_max_index = np.searchsorted(self.RT_inputs['lam_vac_raw'], maximal, side='right')
    
            # Extrahiere die relevanten Daten für measured_data
            lam_vac_RT = self.RT_inputs['lam_vac_raw'][RT_min_index:RT_max_index]
            

            T_data = self.RT_inputs['T_raw'][RT_min_index:RT_max_index]
            R_data = self.RT_inputs['R_raw'][RT_min_index:RT_max_index]
            

            self.RT_inputs['lam_vac'] = lam_vac_RT
            self.RT_inputs['R'] = R_data 
            self.RT_inputs['T'] = T_data
        else:
            pass
    
    def delete_subs_RT(self):
        print('RT substrate nk-data deleted')
        self.RT_upload_nk_check.setVisible(False)
        keys = ['substrate lam_vac', 'M_real', 'M_imag']
        for key in keys:
            if key in self.RT_inputs:
                del self.RT_inputs[key]
            else:
                pass
        self.RT_subs_file.setText('')
    def delete_subs_SE(self):
        keys = ['substrate lam_vac', 'M_real', 'M_imag']
        for key in keys:
            if key in self.SE_inputs:
                del self.SE_inputs[key]
            else:
                pass
        self.SE_upload_nk_check.setVisible(False)
        
        self.SE_subs_file.setText('')

    def delete_RT(self):
        keys = ['lam_vac_raw', 'R_raw', 'T_raw','lam_vac', 'R', 'T']
        for key in keys:
            if key in self.RT_inputs:
                del self.RT_inputs[key]
            else:
                pass
        print('RT data deleted')
        self.RT_upload_check.setVisible(False)
       
        self.RT_file.setText('')
        console_window.update_RT_plot_signal.emit([],[],[],)
    def delete_SE(self):
        keys = ['theta_values',  'lam_vac_raw', 'psis_raw', 'deltas_raw', 'lam_vac_SE','psis_data', 'deltas_data']
        self.SE_upload_check.setVisible(False)
        for key in keys:
            if key in self.SE_inputs:
                del self.SE_inputs[key]
            else:
                pass
        self.SE_file.setText('')
        console_window.update_SE_plot_signal.emit([],[],[],[])

    def interpol_subs(self):
        checkbox_RT_fit_checked = self.main_checkbox_fit_RT.isChecked()
        checkbox_SE_fit_checked = self.main_checkbox_fit_SE.isChecked()
        checkbox_RT_checked = self.main_checkbox_calc_RT.isChecked()
        checkbox_SE_checked = self.main_checkbox_calc_SE.isChecked()
        
        lam_vacsubs_RT = self.RT_inputs.get('substrate lam_vac', None)
        M_real_RT = self.RT_inputs.get('M_real', None)
        M_imag_RT = self.RT_inputs.get('M_imag', None)
        lam_vac_RT = self.RT_inputs.get('lam_vac_raw', None)
        
        lam_vacsubs_SE = self.SE_inputs.get('substrate lam_vac', None)
        M_real_SE = self.SE_inputs.get('M_real', None)
        M_imag_SE = self.SE_inputs.get('M_imag', None)
        lam_vac_SE = self.SE_inputs.get('lam_vac_raw', None)

        self.lam_min_plot = self.model_inputs['lam_min']
        self.lam_max_plot = self.model_inputs['lam_max']
        
        if lam_vac_RT is None:
            lam_vac_RT =  np.linspace(self.lam_min_plot, self.lam_max_plot, 100)
            self.RT_inputs['lam_vac'] = lam_vac_RT
        if lam_vac_SE is None:
            lam_vac_SE =  np.linspace(self.lam_min_plot, self.lam_max_plot, 100)
            self.SE_inputs['lam_vac'] = lam_vac_SE

        if checkbox_RT_fit_checked or checkbox_RT_checked:
            print('interpolated RT substrate')
            
            n_s = interp1d(lam_vacsubs_RT, M_real_RT[:, 1], kind='linear')(lam_vac_RT) 
            k_s = interp1d(lam_vacsubs_RT, M_imag_RT[:, 1], kind='linear')(lam_vac_RT)
            Ns_RT = n_s + 1j* k_s
            subs_RT = []
            subs_RT.append(lam_vac_RT)
            subs_RT.append(Ns_RT)
            if subs_RT is not None:
                self.RT_inputs['subs_RT'] = subs_RT
        if checkbox_SE_fit_checked or checkbox_SE_checked:
            print('interpolated SE substrate')

            n_s = interp1d(lam_vacsubs_SE, M_real_SE[:, 1], kind='linear')(lam_vac_SE) 
            k_s = interp1d(lam_vacsubs_SE, M_imag_SE[:, 1], kind='linear')(lam_vac_SE)
            Ns_SE = n_s + 1j* k_s
            subs_SE = []
            subs_SE.append(lam_vac_SE)
            subs_SE.append(Ns_SE)
            if subs_SE is not None:
                self.SE_inputs['subs_SE'] = subs_SE
    def interpol_data(self):
        
        print('interpolated data')
        checkbox_RT_fit_checked = self.main_checkbox_fit_RT.isChecked()
        checkbox_SE_fit_checked = self.main_checkbox_fit_SE.isChecked()
        
        checkbox_RT_checked = self.main_checkbox_calc_RT.isChecked()
        checkbox_SE_checked = self.main_checkbox_calc_SE.isChecked()
        if (checkbox_RT_checked or checkbox_RT_fit_checked) and (checkbox_SE_checked or checkbox_SE_fit_checked):

            lam_vac_SE_raw = self.SE_inputs.get('lam_vac_raw', None)
            lam_vac_RT_raw = self.RT_inputs.get('lam_vac_raw', None)
            
            lam_vac_SE = self.SE_inputs.get('lam_vac', None)
            lam_vac_RT = self.RT_inputs.get('lam_vac', None)
            
            R_data_raw = self.RT_inputs.get('R_raw', None)
            T_data_raw = self.RT_inputs.get('T_raw', None)
    
            psis_data_raw = self.SE_inputs.get('psis_raw', None)
            deltas_data_raw = self.SE_inputs.get('deltas_raw', None)
            theta_values = self.SE_inputs.get('thetas', None)
            
            R_data = self.RT_inputs.get('R', None)
            T_data = self.RT_inputs.get('T', None)
    
            psis_data = self.SE_inputs.get('psis', None)
            deltas_data = self.SE_inputs.get('deltas', None)
            
    
            len_RT = len(lam_vac_RT) if lam_vac_RT is not None else 0
            len_SE = len(lam_vac_SE) if lam_vac_SE is not None else 0
            
            R_reg = []
            T_reg = []
            psi_reg = []
            delta_reg = []
            lam_vac_all = []
            
            if len_RT < len_SE:
                lam_vac_all = lam_vac_SE
                psi_reg = np.array(psis_data).flatten()
                delta_reg = np.array(deltas_data).flatten()
                if len_RT > 0:
                    print('use SE range')
                    
                    R_reg = interp1d(lam_vac_RT_raw, R_data_raw, kind='linear')(lam_vac_SE) 
                    T_reg = interp1d(lam_vac_RT_raw, T_data_raw, kind='linear')(lam_vac_SE)
                    
                if len_RT == 0:
                    R_reg = None
                    T_reg =  None
                
                    self.model_inputs['lam_examp'] = lam_vac_all
            if len_SE < len_RT:
                lam_vac_all = lam_vac_RT
                R_reg = R_data
                T_reg = T_data
                if len_SE > 0:
                    print('use RT range')

                    psi_reg = []
                    delta_reg = []
                    for i, theta in enumerate(theta_values):
                        psi_interp = interp1d(lam_vac_SE_raw, psis_data_raw[i], kind='linear')
                        delta_interp = interp1d(lam_vac_SE_raw, deltas_data_raw[i], kind='linear')
                        psi_reg.append(psi_interp(lam_vac_RT))
                        delta_reg.append(delta_interp(lam_vac_RT))
                if len_SE == 0:
                    psi_reg = None
                    delta_reg = None
                    
            
            self.RT_inputs['R_reg'] = R_reg
            self.RT_inputs['T_reg'] = T_reg
            self.RT_inputs['lam_vac_reg'] = lam_vac_all
            
            self.SE_inputs['psis_reg']= psi_reg
            self.SE_inputs['deltas_reg']= delta_reg
            self.SE_inputs['lam_vac_reg']= lam_vac_all
                        
    def inputs(self):
        print('read inputs')
        DiE_model = self.list_model.currentText()
        
        if DiE_model == 'Choose a dielectric function model':
           self.error_found = True
           QtWidgets.QMessageBox.warning(self, "Error", 'Please choose a dielectric function model.')



        checkbox_RT_checked = self.main_checkbox_calc_RT.isChecked()
        checkbox_SE_checked = self.main_checkbox_calc_SE.isChecked()
        checkbox_RT_fit_checked = self.main_checkbox_fit_RT.isChecked()
        checkbox_SE_fit_checked = self.main_checkbox_fit_SE.isChecked()
        
        self.model_inputs['RT calc'] = checkbox_RT_checked
        self.model_inputs['SE calc'] = checkbox_SE_checked
        self.model_inputs['RT fit'] = checkbox_RT_fit_checked
        self.model_inputs['SE fit'] = checkbox_SE_fit_checked

        self.model_inputs['ftol'] = float(self.f_tol.text())
        self.model_inputs['xtol'] = float(self.x_tol.text())
        self.model_inputs['gtol'] = float(self.g_tol.text())
        self.model_inputs['threshold'] = float(self.threshold.text())
        
        
        self.model_inputs['Model'] = DiE_model
        d_f =  float(self.d_f_0.text())
        self.model_inputs['model_d_f_0'] = d_f
        
        
        self.model_inputs['model_alpha_R'] = float(self.alpha_R.text())
        self.model_inputs['model_alpha_T'] = float(self.alpha_T.text())
        self.model_inputs['model_alpha_psi'] = float(self.alpha_psi.text())
        self.model_inputs['model_alpha_delta'] = float(self.alpha_delta.text())
        
        self.model_inputs['lam_min'] = float(self.plot_min.text())
        self.model_inputs['lam_max'] = float(self.plot_max.text())
        
        #TL
        n = 0
        d_r =  float(self.d_r.text())
        f_1 =  float(self.f_1.text())
        params_EMA = [d_r, f_1]
        #TLD 
        params = []
        labels = ['ε_inf']
        
        label_d = 'Film thickness'
        label_EMA = ['EMA layer thickness', 'air volumen content']
        try:
            if DiE_model == 'Tauc Lorentz' or DiE_model == 'Tauc Lorentz + Drude':
    
                TL_initial_eps = float(self.TL_eps.text())
    
                TL_inital1 = [float(self.TL1_Eg.text()), 
                              float(self.TL1_C.text()),
                              float(self.TL1_E0.text()),
                              float(self.TL1_A.text())]
                TL_inital2 = [float(self.TL2_Eg.text()), 
                              float(self.TL2_C.text()),
                              float(self.TL2_E0.text()),
                              float(self.TL2_A.text())]
                TL_inital3 = [float(self.TL3_Eg.text()), 
                              float(self.TL3_C.text()),
                              float(self.TL3_E0.text()),
                              float(self.TL3_A.text())]
                TL_inital4 = [float(self.TL4_Eg.text()), 
                              float(self.TL4_C.text()),
                              float(self.TL4_E0.text()),
                              float(self.TL4_A.text())]
                Drude_initial = [float(self.Drude_A.text()), 
                              float(self.Drude_L.text())]
                
                osz = [self.Drude.isChecked(), self.TL_1.isChecked(), self.TL_2.isChecked(), self.TL_3.isChecked(), self.TL_4.isChecked()]
                
                params.append(TL_initial_eps)
                # Verkettung der Oszillatorwerte basierend auf den Checkbox-Zuständen
                label_D = ['A', '$\\Gamma$']
               
                for i in range(5):
                      
                    if i == 0:
                        if osz[i]:
                            params.extend(Drude_initial)
                            labels.extend(label_D)
                    if i > 0:
                        if osz[i]:
                            n += 1
                            params.extend(eval(f"TL_inital{i}"))
                            label_TL = [f'Eg_{i}', f'C_{i}', f'E0_{i}', f'A_{i}']
    
                            labels.extend(label_TL)
                if not any(osz):
                    self.error_found = True
                    QtWidgets.QMessageBox.warning(self, "Error", 'Please choose on of the oszillators for your dielectric function model.')
    
                if self.Bruggeman.isChecked():
                    params.extend(params_EMA)
                    labels.extend(label_EMA)
                    EMA = 'true'
                    print('EMA checked')
                else:
                    EMA = 'false'
                    print('EMA not checked')
                
                
                
            #Cauchy
            if DiE_model == 'Cauchy':
                
                Cauchy_eps = float(self.cauchy_eps.text())
                params.append(Cauchy_eps)
                
                Cauchy_0 = [float(self.cauchy_A.text()), 
                              float(self.cauchy_B.text())]
                Cauchy_1 = float(self.cauchy_C.text())
                Cauchy_2 = float(self.cauchy_D.text())
    
                labels_1 = ['A', 'B']
                labels_23 = ['C', 'D']
                
                params.extend(Cauchy_0)
                labels.extend(labels_1)
                
                terms = [self.cauchy_T2.isChecked(),self.cauchy_T3.isChecked()]
                n = 1
                for i,check in enumerate(terms):
                    if check:
                        n += 1
                        params.append(f"Cauchy_{i+1}")
                        labels.append(labels_23[i])
    
                
            #sellmeier
            if DiE_model == 'Sellmeier':
    
            
                SM_eps = float(self.SM_eps.text())
                params.append(SM_eps)
    
                SM_1 = [float(self.SM_B1.text()), 
                              float(self.SM_C1.text())]
                SM_2 = [float(self.SM_B2.text()), 
                              float(self.SM_C2.text())]
                SM_3 = [float(self.SM_B3.text()), 
                              float(self.SM_C3.text())]
                
                params.extend(SM_1)
            
                terms = [self.SM_T2.isChecked(), self.SM_T3.isChecked()]
                
                n = 1
                for i,check in enumerate(terms):
                    if check:
                        n += 1
                        params.extend(eval(f"SM_{i}"))
                        label_SM = [f'B_{i}', f'C_{i}']
                        labels.extend(label_SM)
                
            params.append(d_f)
            labels.append(label_d)
            if self.Bruggeman.isChecked():
                
                params.extend(params_EMA)
                labels.extend(label_EMA)
                EMA = 'true'
                print('EMA checked')
                if f_1 > 1:
                    self.error_found = True
                    QtWidgets.QMessageBox.warning(self, "Error", 'The fraction of air in the Bruggeman roughness layer needs to be <= 1.')
    
            else:
                EMA = 'false'
                print('EMA not checked')
            
            self.model_inputs[DiE_model] = {'n_osz' : n,
                                   'params' : params,
                                   'EMA': EMA, 'labels': labels
                                   }
            
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"An error occurred: {e}")
            
    def inputs_RT(self):
        print('read RT inputs')


        self.RT_inputs['d_s_subs'] = float(self.RT_d_2.text())
            
        self.RT_inputs['theta_0_R'] =  float(self.RT_angle_R.text())
        self.RT_inputs['theta_0_T'] = float(self.RT_angle_T.text())
        self.RT_inputs['minimal'] = float(self.RT_min.text())
        self.RT_inputs['maximal'] = float(self.RT_max.text())

        self.RT_inputs['subs'] = self.RT_substrate.currentText()
        
        self.RT_inputs['c_subs'] = self.RT_type_subs.currentText()
        if self.RT_inputs['subs'] != 'Other (upload nk file)':
            lam_vacsubs_values, M_real, M_imag = ld.load_subs(self.RT_inputs['subs'])
            self.RT_inputs['substrate lam_vac'] = lam_vacsubs_values
            self.RT_inputs['M_real'] = M_real
            self.RT_inputs['M_imag'] = M_imag            
    def inputs_SE(self):
        print('read SE inputs')
        self.SE_inputs['minimal'] = float(self.SE_min.text())
        self.SE_inputs['maximal'] = float(self.SE_max.text())

        self.SE_inputs['subs'] = self.SE_substrate.currentText()
        
        theta_min = float(self.SE_angle_min.text())
        theta_max = float(self.SE_angle_max.text())
        dt = float(self.SE_angle_width.text())
        self.SE_inputs['thetas'] = np.arange(theta_min, theta_max+dt, dt)
        
        if self.SE_inputs['subs'] != 'Other (upload nk file)':
            lam_vacsubs_values, M_real, M_imag = ld.load_subs(self.SE_inputs['subs'])
            self.SE_inputs['substrate lam_vac'] = lam_vacsubs_values
            self.SE_inputs['M_real'] = M_real
            self.SE_inputs['M_imag'] = M_imag
    
    def HW_RT(self):

        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText('The File has to be a .csv File with three columns:<br><br><i>wavelength [nm] ; T [-] ; R [-]</i><br><br> No Headline. ";" as separator.')
        Help.exec()
    def HW_SE(self):

        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText( 'The File has to be a .csv File that contains the SE-Data for all measured incident angles. '
    'The measured angles should be written in the first row, i.e. <br><br> '
    '<i>50;60;..</i><br><br> '
    'From row 2, the Data should contain the actual data:<br><br> '
    '<i>wavelength [nm] ; ψ [rad] 50°; Δ [rad] 50°; ψ [rad] 60°; Δ [rad] 60°; ...</i><br><br> '
    'No headline. ";" as separator.')
        Help.exec()
    def HW_RT_tmm(self):
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText('The reflection and transmission are calculated with an '
                     'open source transfer-matrix package '
                     'named "tmm" written by Steven Byrnes, http://sjbyrnes.com')

        Help.exec()
    def HW_SE_tmm(self):
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText('The ellipsometry angles are calculated with an '
                     'open source transfer-matrix package '
                     'named "tmm" written by Steven Byrnes, http://sjbyrnes.com')

        Help.exec()
    def HW_subs(self):
       
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText( 'The File has to be a .csv File that contains the nk-Data for the substrate.  '
    'It should containe three columns: <br><br> '
    '<i>wavelength [nm]; n ; k</i><br><br> '
    )
        Help.exec()
    def HW_model(self):

        help_window = QtWidgets.QWidget()
        help_window.setWindowTitle('Help')
        
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel()
        
        # Bildpfad angeben
        pixmap = QtGui.QPixmap('pictures\Model.png')  # Ersetzen Sie 'path_to_your_image.png' durch den Pfad zu Ihrem Bild
        label.setPixmap(pixmap)
        
        layout.addWidget(label)
        help_window.setLayout(layout)
        
        help_window.show()
        self.help_window = help_window
    def HW_TL(self):
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText('The exampled values are the parameters of nc-SiOx.')

        Help.exec()
    def HW_cauchy(self):
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText('The exampled values are the parameters of nc-SiOx in the IR spectre.')

        Help.exec()
    def HW_SM(self):
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText('The exampled values are the parameters of borsicate glass.')

        Help.exec()
    def HW_angle(self):
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText( 'These entrys are ignored if a SE-Data file is uploaded.'
                     'This is only used for calculating SE-Spectres from given dielectric function parameters from the tab "Model"'
    )
        Help.exec()
    def HW_tol(self):
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText('The File has to be a .csv File with three columns:<br><br><i>wavelength ; T ; R</i><br><br>wavelength in nm, R and T as fractions. ";" as separator.')
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText('Termination conditions. According to leastsquare lm-optimization python module.\n '
                     'ftol = Tolerance of changes in Error function \n '
                     'xtol = Tolerance of changes in parameter\n '
                     'gtol = Tolerance of norm of the gradient\n')
        Help.exec()
        
    def HW_tresh(self):
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText('The File has to be a .csv File with three columns:<br><br><i>wavelength ; T ; R</i><br><br>wavelength in nm, R and T as fractions. ";" as separator.')
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText( 'For numerical security. '
                     'Neighbouring SE measurement points which are differing more than this threshold '
                     'will not be taken into account in error calculation.'
    )
        Help.exec()
            
    def HW_error(self):
        Help = QtWidgets.QMessageBox()
        Help.setWindowTitle('Help')
        Help.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        Help.setText('To focus on a certain measurement choose ' 
                     'a error weighting > 1. Also you should consider ' 
                     'the amount of data you have, i.e. if you are fitting RT ' 
                     'and SE simultaniously and you are considering 3 SE measurements ' 
                     'for different incident angles, you should adapt the error weighting '
                     'for R and T to 3. \n \n The number of measurement points is adapted automatically ' 
                     'for simultanious RT and SE fitting and does\'nt need to be considered here.')

        Help.exec()
    def Helpbook(self):
        print('open Helpbook')
        pdf_path = r'pdf for python\Helpbook.pdf' # Change this to the path of your PDF file
        if sys.platform == "win32":
            os.startfile(pdf_path)
        elif sys.platform == "darwin":
            os.system(f"open '{pdf_path}'")
        else:
            os.system(f"xdg-open '{pdf_path}'")         
    def Errors(self):
        RT_fit = self.model_inputs['RT fit']
        SE_fit = self.model_inputs['SE fit']
        RT_calc = self.model_inputs['RT calc']
        SE_calc = self.model_inputs['SE calc']

        self.error_found = False

        if RT_fit:
            key = 'R'
            if key not in self.RT_inputs:
                print('no RT data')
                self.error_found = True
                QtWidgets.QMessageBox.warning(self, "Error", 'Please upload your RT-measurements to proceed RT fitting.')

        if RT_fit or RT_calc:
            key = 'substrate lam_vac'
            if key not in self.RT_inputs:
                print('no substrate chosen')
                self.error_found = True
                QtWidgets.QMessageBox.warning(self, "Error", 'Please choose the substrate of your RT measurement or upload substrate nk-Data.')

        if SE_fit:
            key = 'psis'
            if key not in self.SE_inputs:
                print('no SE data')
                self.error_found = True
                QtWidgets.QMessageBox.warning(self, "Error", 'Please upload your SE-measurements to proceed RT fitting.')

        if SE_fit or SE_calc:
            key = 'substrate lam_vac'
            if key not in self.SE_inputs:
                print('no substrate chosen')
                self.error_found = True
                QtWidgets.QMessageBox.warning(self, "Error", 'Please choose the substrate of your SE measurement or upload substrate nk-Data.')
                
        if not self.error_found:
            print('Error found', self.error_found)
            self.interpol_data()
            self.interpol_subs()
            self.safety_check()
    def start_simulation(self):
        # Delete the old worker and thread if they exist
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None

        # Create a new worker and thread
        self.thread = QThread()
        self.worker = OptimizerWorker(self.model_inputs, self.RT_inputs, self.SE_inputs)
       
        
        self.worker.finished.connect(self.thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater)
        
        if self.model_inputs['RT fit'] and not self.model_inputs['SE fit']:
            self.worker.iteration_update.connect(console_window.display_RT_plot)
        if self.model_inputs['SE fit'] and not self.model_inputs['RT fit']:
            self.worker.iteration_update.connect(console_window.display_SE_plot)
        if self.model_inputs['RT fit'] and self.model_inputs['SE fit']:
            self.worker.iteration_update_all.connect(console_window.display_RTSE_plot)
            
        self.worker.result_ready.connect(self.print_result)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.thread.start()
    
    def safety_check(self):
        if self.count > 1:
            self.close_result()
        RT_fit = self.model_inputs['RT fit']
        SE_fit = self.model_inputs['SE fit']
        RT_calc = self.model_inputs['RT calc']
        SE_calc = self.model_inputs['SE calc'] 
        Simulation = QtWidgets.QMessageBox()
        Simulation.setWindowTitle('Proceed')
        Simulation.setText('Are your sure you want to proceed?')
        Simulation.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        result_settings= Simulation.exec()
        if result_settings == QtWidgets.QMessageBox.Yes:
            self.count += 1

            if RT_fit or SE_fit:
                self.start_simulation()
            if RT_calc or SE_calc:
                if not RT_fit and not SE_fit:
                    self.print_result({})
            if not RT_fit and not SE_fit and not RT_calc and not SE_calc:
                self.print_result({})

                 
    def print_result(self, result):
        Calculator  = Calculation(self.model_inputs, self.RT_inputs, self.SE_inputs, result)
        Calculator.calculation()
        print('calculation succeeded')

        self.show_results = Result(self.model_inputs, self.SE_inputs, self.RT_inputs, Calculator.results)
        self.show_results.show_results()
        
        RT_fit = self.model_inputs['RT fit']
        SE_fit = self.model_inputs['SE fit']
        RT_calc = self.model_inputs['RT calc']
        SE_calc = self.model_inputs['SE calc']        
        if RT_fit or RT_calc:
            if not SE_fit and not SE_calc:
                print('show RT results')
                self.show_results.plot_RT()
            if SE_fit or SE_calc:
                print('show RT+SE results')
                self.show_results.plot_RTSE()
        if SE_fit or SE_calc:
            if not RT_fit and not RT_calc:
                print('show SE results')
                self.show_results.plot_SE()
            if RT_fit or RT_calc:
                print('show SE+RT results')

                self.show_results.plot_RTSE()
        if not RT_fit and not RT_calc and not SE_fit and not SE_calc:
            print('show nk plot')
            self.show_results.plot_nk()
        self.show_results.show()

class Optimizer():

    def __init__(self, model_inputs, SE_inputs, RT_inputs):
        
        self.model_inputs = model_inputs
        self.SE_inputs = SE_inputs
        self.RT_inputs = RT_inputs
        
        console_window.stopButton.clicked.connect(self.running_status)
        self._is_running = True

        self.results = {}
        self.count = 0 
        
    def running_status(self):
        print('optimizer stopped')
        self._is_running = False
        
    def TLD_SE_rough(self, params, *args):
       
        while self._is_running:
            self.count +=1
            (n_osz, subs_SE,
                       theta_values, lam_vac_SE, psis_data, deltas_data,
                       alpha_psi, alpha_delta, callback) = args
    
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_rough(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_SE]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
          
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
            
            if self.count % 10 == 0:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
            
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
                return np.array([])
            
            return np.concatenate((alpha_psi*Error_psi,
                              alpha_delta*Error_delta_filtered))
    def TLD_SE(self, params, *args):
        while self._is_running:
            self.count +=1
            (n_osz, subs_SE,
                       theta_values, lam_vac_SE, psis_data, deltas_data,
                       alpha_psi, alpha_delta, callback) = args
                       
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_SE]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
          
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            # Berechne die summierte MSE ohne die entfernten Messwerte
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
            
            if self.count % 10 == 0:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
            
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
                return np.array([])
            return np.concatenate((alpha_psi*Error_psi,
                              alpha_delta*Error_delta_filtered))
            
    def TLD_RT_rough(self, params,*args):
        while self._is_running:
            self.count += 1
            n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, lam_vac_RT, R_data, T_data, subs_RT, alpha_R, alpha_T, callback = args
            R, T = np.array([RT_rough(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_RT]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_RT))
            Error_T = (T - T_data) * 100 / max_T / np.sqrt(len(lam_vac_RT))
            MSE = np.sqrt(((R - R_data)**2 + (T - T_data)**2).mean()) * 100
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            if self.count % 10 == 0:
    
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
                return np.array([])
            return np.concatenate((alpha_R*Error_R, alpha_T*Error_T))
    
    def TLD_RT(self, params,*args):
        while self._is_running:
            self.count +=1
            n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, lam_vac_RT, R_data, T_data, subs_RT, alpha_R, alpha_T, callback = args
            R, T = np.array([RT(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_RT]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_RT))
            Error_T = (T - T_data) * 100 / max_T / np.sqrt(len(lam_vac_RT))
            MSE = np.sqrt(((R - R_data)**2 + (T - T_data)**2).mean()) * 100
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            if self.count % 10 == 0:
    
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
                return np.array([])

            return np.concatenate((alpha_R*Error_R, alpha_T*Error_T))
    
    def TLD_all_rough(self, params, *args):
        while self._is_running:
        
            self.count +=1
            (n_osz, d_s_RT,
             theta_0_T, theta_0_R,
             c_list,
             R_data, T_data,
             subs_RT,
             theta_values,
             lam_vac_all, psis_data, deltas_data,
             subs_SE,
             alpha_R, alpha_T, alpha_psi, alpha_delta, callback) = args
    
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_rough(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_all]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
            R_values, T_values = np.array([RT_rough(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_all]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R_values - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_all))
            Error_T = (T_values - T_data) * 100 / max_T / np.sqrt(len(lam_vac_all))
            
            MSE_RT = np.sqrt(((R_values - R_data)**2 + (T_values - T_data)**2).mean()) * 100
    
            
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            # Berechne die summierte MSE ohne die entfernten Messwerte
            if self.count % 10 == 0:
                
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
                return np.array([])
        return np.concatenate((alpha_R*Error_R,
                              alpha_T*Error_T,
                              alpha_psi*Error_psi,
                              alpha_delta*Error_delta_filtered))
    def TLD_all(self, params, *args):
        while self._is_running:
            
            self.count += 1
            (n_osz, d_s_RT,
             theta_0_T, theta_0_R,
             c_list,
             R_data, T_data,
             subs_RT,
             theta_values,
             lam_vac_all, psis_data, deltas_data,
             subs_SE,
             alpha_R, alpha_T, alpha_psi, alpha_delta, callback) = args
            
            psis = []
            deltas = []
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_all]).T
               psis.append(psi_values)
               deltas.append(delta_values)
    
            R_values, T_values = np.array([RT(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_all]).T
            
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R_values - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_all))
            Error_T = (T_values - T_data) * 100 / max_T / np.sqrt(len(lam_vac_all))
            
            MSE_RT = np.sqrt(((R_values - R_data)**2 + (T_values - T_data)**2).mean()) * 100
    
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            # Berechne die summierte MSE ohne die entfernten Messwert
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            # Berechne die summierte MSE ohne die entfernten Messwerte
            if self.count % 10 == 0:
                
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
                return np.array([])
        
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T,
                                  alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
    
    def TL_all_rough(self, params, *args):
        while self._is_running:
            self.count +=1
            (n_osz, d_s_RT,
             theta_0_T, theta_0_R,
             c_list,
             R_data, T_data,
             subs_RT,
             theta_values,
             lam_vac_all, psis_data, deltas_data,
             subs_SE,
             alpha_R, alpha_T, alpha_psi, alpha_delta, callback ) = args
    
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_rough_TL(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_all]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
            R_values, T_values = np.array([RT_rough_TL(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_all]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R_values - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_all))
            Error_T = (T_values - T_data) * 100 / max_T / np.sqrt(len(lam_vac_all))
            
            MSE_RT = np.sqrt(((R_values - R_data)**2 + (T_values - T_data)**2).mean()) * 100
    
            
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            # Berechne die summierte MSE ohne die entfernten Messwerte
            if self.count % 10 == 0:
                
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values, T_values)
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values, T_values)
                return np.array([])
        
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T,
                                  alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
    def TL_all(self, params, *args):
        while self._is_running:
            self.count += 1
            (n_osz, d_s_RT,
            theta_0_T, theta_0_R,
            c_list,
            R_data, T_data,
            subs_RT,
            theta_values,
            lam_vac_all, psis_data, deltas_data,
            subs_SE,
            alpha_R, alpha_T, alpha_psi, alpha_delta, callback) = args
            
            psis = []
            deltas = []
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_TL(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_all]).T
               psis.append(psi_values)
               deltas.append(delta_values)
    
            R_values, T_values = np.array([RT_TL(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_all]).T
            
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R_values - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_all))
            Error_T = (T_values - T_data) * 100 / max_T / np.sqrt(len(lam_vac_all))
            
            MSE_RT = np.sqrt(((R_values - R_data)**2 + (T_values - T_data)**2).mean()) * 100
    
    
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            # Berechne die summierte MSE ohne die entfernten Messwerte
            if self.count % 10 == 0:
                
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
                return np.array([])
        
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T,
                                  alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
    def TL_SE_rough(self, params, *args):
        while self._is_running:
            
            self.count +=1
            (n_osz, subs_SE,
                       theta_values, lam_vac_SE, psis_data, deltas_data,
                       alpha_psi, alpha_delta, callback) = args
                       
            n_osz, subs_SE, 
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_rough_TL(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_SE]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
          
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
            
            if self.count % 10 == 0:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
                return np.array([])
            return np.concatenate((alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
    def TL_SE(self, params, *args):
        while self._is_running:
            self.count +=1
            (n_osz, subs_SE,
                       theta_values, lam_vac_SE, psis_data, deltas_data,
                       alpha_psi, alpha_delta, callback) = args
                       
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_TL(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_SE]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
          
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
            
            if self.count % 10 == 0:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
                
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)

                return np.array([])

            return np.concatenate((alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
    def TL_RT_rough(self, params,*args):
        while self._is_running:
            self.count += 1
            n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, lam_vac_RT, R_data, T_data, subs_RT, alpha_R, alpha_T, callback = args
            R, T = np.array([RT_rough_TL(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_RT]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_RT))
            Error_T = (T - T_data) * 100 / max_T / np.sqrt(len(lam_vac_RT))
            MSE = np.sqrt(((R - R_data)**2 + (T - T_data)**2).mean()) * 100
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            if self.count % 10 == 0:
    
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
                return np.array([])
            return np.concatenate((alpha_R*Error_R, alpha_T*Error_T))
    def TL_RT(self, params,*args):
        while self._is_running:
            self.count +=1
            n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, lam_vac_RT, R_data, T_data, subs_RT, alpha_R, alpha_T, callback  = args
            
            
                
            R, T = np.array([RT_TL(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_RT]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_RT))
            Error_T = (T - T_data) * 100 / max_T / np.sqrt(len(lam_vac_RT))
            MSE = np.sqrt(((R - R_data)**2 + (T - T_data)**2).mean()) * 100
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            if self.count % 10 == 0:
    
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())

                return np.array([])
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T))
    
    def C_all_rough(self, params, *args):
        while self._is_running:
            self.count +=1
            (n_osz, d_s_RT,
             theta_0_T, theta_0_R,
             c_list,
             R_data, T_data,
             subs_RT,
             theta_values,
             lam_vac_all, psis_data, deltas_data,
             subs_SE,
             alpha_R, alpha_T, alpha_psi, alpha_delta, callback) = args
    
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_rough_C(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_all]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
            R_values, T_values = np.array([RT_rough_C(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_all]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R_values - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_all))
            Error_T = (T_values - T_data) * 100 / max_T / np.sqrt(len(lam_vac_all))
            
            MSE_RT = np.sqrt(((R_values - R_data)**2 + (T_values - T_data)**2).mean()) * 100
    
            
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            # Berechne die summierte MSE ohne die entfernten Messwerte
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            # Berechne die summierte MSE ohne die entfernten Messwerte
            if self.count % 10 == 0:
                
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())

                return np.array([])
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T,
                                  alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
    def C_all(self, params, *args):
        while self._is_running:
            (n_osz, d_s_RT,
             theta_0_T, theta_0_R,
             c_list,
             R_data, T_data,
             subs_RT,
             theta_values,
             lam_vac_all, psis_data, deltas_data,
             subs_SE,
             alpha_R, alpha_T, alpha_psi, alpha_delta, callback) = args
            
            psis = []
            deltas = []
            print('calculation SE')
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_C(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_all]).T
               psis.append(psi_values)
               deltas.append(delta_values)
            print('calculation RT')
    
            R_values, T_values = np.array([RT_C(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_all]).T
            
            print('calculate RT Error')
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R_values - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_all))
            Error_T = (T_values - T_data) * 100 / max_T / np.sqrt(len(lam_vac_all))
            
            MSE_RT = np.sqrt(((R_values - R_data)**2 + (T_values - T_data)**2).mean()) * 100
    
            print('calculate SE Error')
    
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
           
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            # Berechne die summierte MSE ohne die entfernten Messwerte
            if self.count % 10 == 0:
                
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())

                return np.array([])
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T,
                                  alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
    def C_SE_rough(self, params, *args):
        while self._is_running:
            self.count +=1
            (n_osz, subs_SE,
                       theta_values, lam_vac_SE, psis_data, deltas_data,
                       alpha_psi, alpha_delta, callback) = args
                       
            n_osz, subs_SE, 
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_rough_C(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_SE]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
          
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            # Berechne die summierte MSE ohne die entfernten Messwerte
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
            
            if self.count % 10 == 0:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
            
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)

                return np.array([])
    
            return np.concatenate((alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
    def C_SE(self, params, *args):
        while self._is_running:
            self.count +=1
            (n_osz, subs_SE,
                       theta_values, lam_vac_SE, psis_data, deltas_data,
                       alpha_psi, alpha_delta, callback) = args
                       
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_C(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_SE]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
          
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]
            # Berechne die summierte MSE ohne die entfernten Messwerte
            if self.count % 10 == 0:
            
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
            
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
                return np.array([])

            return np.concatenate((alpha_psi*Error_psi,
                                   alpha_delta*Error_delta_filtered))
    def C_RT_rough(self, params,*args):
        while self._is_running:
            self.count += 1
            n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, lam_vac_RT, R_data, T_data, subs_RT, alpha_R, alpha_T, callback = args
            R, T = np.array([RT_rough_C(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_RT]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_RT))
            Error_T = (T - T_data) * 100 / max_T / np.sqrt(len(lam_vac_RT))
            MSE = np.sqrt(((R - R_data)**2 + (T - T_data)**2).mean()) * 100
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            if self.count % 10 == 0:
    
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())

                return np.array([])
    
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T))
    def C_RT(self, params,*args):
        while self._is_running:
            self.count +=1
            n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, lam_vac_RT, R_data, T_data, subs_RT, alpha_R, alpha_T, callback = args
            R, T = np.array([RT_C(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_RT]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_RT))
            Error_T = (T - T_data) * 100 / max_T / np.sqrt(len(lam_vac_RT))
            MSE = np.sqrt(((R - R_data)**2 + (T - T_data)**2).mean()) * 100
    
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            if self.count % 10 == 0:
    
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
    
                return np.array([])
                
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T))
    
    def SM_all_rough(self, params, *args):
        while self._is_running:
            self.count +=1
            (n_osz, d_s_RT,
             theta_0_T, theta_0_R,
             c_list,
             R_data, T_data,
             subs_RT,
             theta_values,
             lam_vac_all, psis_data, deltas_data,
             subs_SE,
             alpha_R, alpha_T, alpha_psi, alpha_delta, callback) = args
    
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_rough_SM(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_all]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
            R_values, T_values = np.array([RT_rough_SM(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_all]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R_values - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_all))
            Error_T = (T_values - T_data) * 100 / max_T / np.sqrt(len(lam_vac_all))
            
            MSE_RT = np.sqrt(((R_values - R_data)**2 + (T_values - T_data)**2).mean()) * 100
    
            
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            # Berechne die summierte MSE ohne die entfernten Messwerte
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            # Berechne die summierte MSE ohne die entfernten Messwerte
            if self.count % 10 == 0:
                
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
    
                return np.array([])
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T,
                                  alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
        
    def SM_all(self, params, *args):
        while self._is_running:
            self.count += 1
    
            (n_osz, d_s_RT,
             theta_0_T, theta_0_R,
             c_list,
             R_data, T_data,
             subs_RT,
             theta_values,
             lam_vac_all, psis_data, deltas_data,
             subs_SE,
             alpha_R, alpha_T, alpha_psi, alpha_delta, callback) = args
            
            psis = []
            deltas = []
            print('calculation SE')
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_SM(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_all]).T
               psis.append(psi_values)
               deltas.append(delta_values)
            print('calculation RT')
    
            R_values, T_values = np.array([RT_SM(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_all]).T
            
            print('calculate RT Error')
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R_values - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_all))
            Error_T = (T_values - T_data) * 100 / max_T / np.sqrt(len(lam_vac_all))
            
            MSE_RT = np.sqrt(((R_values - R_data)**2 + (T_values - T_data)**2).mean()) * 100
    
            print('calculate SE Error')
    
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            # Berechne die summierte MSE ohne die entfernten Messwerte
    
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            # Berechne die summierte MSE ohne die entfernten Messwerte
            if self.count % 10 == 0:
                
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_RT, MSE_SE_fl, lam_vac_all.tolist(), psis, deltas, R_values.tolist(), T_values.tolist())
    
                return np.array([])
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T,
                                  alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
    def SM_SE_rough(self, params, *args):
            while self._is_running:
                self.count +=1
                (n_osz, subs_SE,
                           theta_values, lam_vac_SE, psis_data, deltas_data,
                           alpha_psi, alpha_delta, callback) = args
                           
                n_osz, subs_SE, 
                psis = []
                deltas = []
        
                for theta in theta_values:
        
                   psi_values, delta_values = np.array([SE_rough_SM(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_SE]).T
                   psis.append(psi_values)
                   deltas.append(delta_values)
                    
              
                psis_fl= np.array(psis).flatten()
                deltas_fl= np.array(deltas).flatten()
                
                Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
                Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
                
                threshold = self.model_inputs['threshold']
        
                # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
                indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
               
                # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
                indices_to_remove = []
                
                # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
                for index in indices:
                    indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
                
                # Entferne Duplikate und sortiere die Liste der Indizes
                indices_to_remove = sorted(list(set(indices_to_remove)))
                #print(indices_to_remove)
        
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
                MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
                
                # Berechne die summierte MSE ohne die entfernten Messwerte
                Model = self.model_inputs['Model']
                labels = self.model_inputs[Model]['labels']
                labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
                
                if self.count % 10 == 0:
                    callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
                if not self._is_running:
                    callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
                
                    return np.array([])
        
                return np.concatenate((alpha_psi*Error_psi,
                                      alpha_delta*Error_delta_filtered))
    def SM_SE(self, params, *args):
        while self._is_running:
            self.count +=1
            (n_osz, subs_SE,
                       theta_values, lam_vac_SE, psis_data, deltas_data,
                       alpha_psi, alpha_delta, callback) = args
                       
            psis = []
            deltas = []
    
            for theta in theta_values:
    
               psi_values, delta_values = np.array([SE_SM(lam_vac, theta, params, (n_osz, subs_SE)) for lam_vac in lam_vac_SE]).T
               psis.append(psi_values)
               deltas.append(delta_values)
                
          
            psis_fl= np.array(psis).flatten()
            deltas_fl= np.array(deltas).flatten()
            
            Errors_psi = (psis_fl - psis_data) / np.sqrt(len(psis_data))
            Errors_delta = (deltas_fl - deltas_data)/ np.sqrt(len(deltas_data))
            
            threshold = self.model_inputs['threshold']
    
            # Finde die Indizes im psi_data-Array, an denen der Unterschied größer als der Schwellenwert ist
            indices = np.where(np.abs(np.diff(deltas_data)) > threshold)[0]
           
            # Erstelle eine Liste der Indizes der Messwerte, die entfernt werden sollen
            indices_to_remove = []
            
            # Füge die Indizes der Messwerte, die entfernt werden sollen, und ihre benachbarten Indizes hinzu
            for index in indices:
                indices_to_remove.extend([index-2, index-1, index, index+1, index+2])
            
            # Entferne Duplikate und sortiere die Liste der Indizes
            indices_to_remove = sorted(list(set(indices_to_remove)))
            #print(indices_to_remove)
    
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
            MSE_SE_fl = np.sqrt((np.concatenate(((Error_psi)**2, (Errors_delta_filtered)**2))).mean()) * 180 / np.pi
            
            # Berechne die summierte MSE ohne die entfernten Messwerte
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
            
            if self.count % 10 == 0:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE_SE_fl, lam_vac_SE.tolist(), psis, deltas)
            
                return np.array([])
            return np.concatenate((alpha_psi*Error_psi,
                                  alpha_delta*Error_delta_filtered))
    def SM_RT_rough(self, params,*args):
        while self._is_running:
            self.count += 1
            n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, lam_vac_RT, R_data, T_data, subs_RT, alpha_R, alpha_T, callback = args
            R, T = np.array([RT_rough_SM(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_RT]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_RT))
            Error_T = (T - T_data) * 100 / max_T / np.sqrt(len(lam_vac_RT))
            MSE = np.sqrt(((R - R_data)**2 + (T - T_data)**2).mean()) * 100
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            if self.count % 10 == 0:
    
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
                return np.array([])

            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T))
    def SM_RT(self, params,*args):
        while self._is_running:
            self.count +=1
            n_osz, d_s_RT, theta_0_T, theta_0_R, c_list, lam_vac_RT, R_data, T_data, subs_RT, alpha_R, alpha_T, callback = args
            R, T = np.array([RT_SM(lam_vac, params, (n_osz, d_s_RT,
                                                             theta_0_T, theta_0_R,
                                                             c_list, subs_RT)) for lam_vac in lam_vac_RT]).T
                
            max_R = R_data.max()
            max_T = T_data.max()
            
            Error_R = (R - R_data) * 100 / max_R  / np.sqrt(len(lam_vac_RT))
            Error_T = (T - T_data) * 100 / max_T / np.sqrt(len(lam_vac_RT))
            MSE = np.sqrt(((R - R_data)**2 + (T - T_data)**2).mean()) * 100
            
            Model = self.model_inputs['Model']
            labels = self.model_inputs[Model]['labels']
            labeled_params = [f'{label} = {param}' for label, param in zip(labels, params)]        
    
            if self.count % 10 == 0:
    
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
            if not self._is_running:
                callback.emit(self.count, labeled_params, MSE, lam_vac_RT.tolist(), R.tolist(), T.tolist())
                return np.array([])
            
            return np.concatenate((alpha_R*Error_R,
                                  alpha_T*Error_T))
    
    def model(self, callback):
        print('simulation starts')

        checkbox_RT_fit_checked = self.model_inputs['RT fit']     
        checkbox_SE_fit_checked = self.model_inputs['SE fit']
        
        Model = self.model_inputs['Model']
        n_osz = self.model_inputs[Model]['n_osz']      
        EMA = self.model_inputs[Model]['EMA']    
        ftol = self.model_inputs['ftol']
        xtol = self.model_inputs['xtol']
        gtol = self.model_inputs['gtol']

        lam_vac_RT = self.RT_inputs.get('lam_vac', None)
        R_data = self.RT_inputs.get('R', None)
        T_data = self.RT_inputs.get('T', None)
        
        lam_vac_all = self.RT_inputs.get('lam_vac_reg', None)
        R_reg = self.RT_inputs.get('R_reg', None)
        T_reg = self.RT_inputs.get('T_reg', None)
    
        lam_vac_SE = self.SE_inputs.get('lam_vac', None)
        psis_data = self.SE_inputs.get('psis', None)
        deltas_data = self.SE_inputs.get('deltas', None)
        theta_values = self.SE_inputs.get('thetas', None)
        
        psi_reg = self.SE_inputs.get('psis_reg', None)
        delta_reg = self.SE_inputs.get('deltas_reg', None)

        alpha_R = self.model_inputs.get('model_alpha_R', None)
        alpha_T = self.model_inputs.get('model_alpha_T', None)
        alpha_psi = self.model_inputs.get('model_alpha_psi', None)
        alpha_delta = self.model_inputs.get('model_alpha_delta', None)
        
        subs_RT = self.RT_inputs.get('subs_RT', None)
        subs_SE = self.SE_inputs.get('subs_SE', None)

        theta_0_R = self.RT_inputs.get('theta_0_R', None)
        theta_0_T = self.RT_inputs.get('theta_0_T', None)
        d_s_RT = self.RT_inputs.get('d_s_subs', None)
    
        self.lam_min_plot = self.model_inputs.get('model_plot_min', None)
        self.lam_max_plot = self.model_inputs.get('model_plot_max', None)
        
        subs_type = self.RT_inputs['c_subs']

        if subs_type == 'incoherent':
            print('incoherent substrate')
            if EMA == 'true':
                c_list = ['i', 'c', 'c', 'i','i']
            elif EMA == 'false':
                c_list = ['i', 'c', 'i','i']
        else:
            print('coherent substrate')
            if EMA == 'true':
                c_list = ['i', 'c', 'c', 'c','i']
            elif EMA == 'false':
                c_list = ['i', 'c', 'c','i']
                
        len_RT = len(lam_vac_RT) if lam_vac_RT is not None else 0
        len_SE = len(lam_vac_SE) if lam_vac_SE is not None else 0
        
        initial_guess = self.model_inputs[Model]['params']
        
        if checkbox_RT_fit_checked and checkbox_SE_fit_checked:
            
                    
            print('fit RT and SE start')
            args = (n_osz, d_s_RT,
             theta_0_T, theta_0_R,
             c_list,
             R_reg, T_reg,
             subs_RT,
             theta_values,
             lam_vac_all, psi_reg, delta_reg,
             subs_SE,
             alpha_R, alpha_T, alpha_psi, alpha_delta, callback)
             
            if Model == 'Tauc Lorentz + Drude':
                if EMA == 'true':
                    print('Fitting all with EMA TL + Drude')
                    opt_param = least_squares(self.TLD_all_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting all TL + Drude')

                    opt_param = least_squares(self.TLD_all,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm',
                                              ftol = ftol, xtol = xtol, gtol =gtol)  
            if Model == 'Tauc Lorentz':
                if EMA == 'true':
                    print('Fitting all with EMA TL')
                    opt_param = least_squares(self.TL_all_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting all TL')

                    opt_param = least_squares(self.TL_all,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol)
            if Model == 'Cauchy':
                if EMA == 'true':
                    print('Fitting all with EMA Cauchy')
                    opt_param = least_squares(self.C_all_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting all Cauchy')

                    opt_param = least_squares(self.C_all,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol)
            if Model == 'Sellmeier':
                if EMA == 'true':
                    print('Fitting all with EMA Sellmeier')
                    opt_param = least_squares(self.SM_all_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting all Sellmeier')

                    opt_param = least_squares(self.SM_all,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol)
            
        if checkbox_RT_fit_checked and not checkbox_SE_fit_checked:
            args = (n_osz, d_s_RT,
             theta_0_T, theta_0_R,
             c_list,
             lam_vac_RT, R_data, T_data,
             subs_RT,
             alpha_R, alpha_T, callback)
            if Model == 'Tauc Lorentz + Drude':
                if EMA == 'true':
                    print('Fitting RT with EMA TL + Drude')
                    opt_param = least_squares(self.TLD_RT_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting RT TL + Drude')

                    opt_param = least_squares(self.TLD_RT,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
            if Model == 'Tauc Lorentz':
                if EMA == 'true':
                    print('Fitting RT with EMA TL')
                    opt_param = least_squares(self.TL_RT_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting RT TL')                  
                   
                    opt_param = least_squares(self.TL_RT,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol)
                    
                    
                        
            if Model == 'Cauchy':
                if EMA == 'true':
                    print('Fitting RT with EMA Cauchy')
                    opt_param = least_squares(self.C_RT_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting RT Cauchy')

                    opt_param = least_squares(self.C_RT,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol)
            if Model == 'Sellmeier':
                if EMA == 'true':
                    print('Fitting RT with EMA Sellmeier')
                    opt_param = least_squares(self.SM_RT_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting RT Sellmeier')

                    opt_param = least_squares(self.SM_RT,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol)
        if checkbox_SE_fit_checked and not checkbox_RT_fit_checked:
            args = (n_osz, subs_SE,
                       theta_values, lam_vac_SE, np.array(psis_data).flatten(), 
                       np.array(deltas_data).flatten(),
                       alpha_psi, alpha_delta, callback)
            if Model == 'Tauc Lorentz + Drude':
                if EMA == 'true':
                    print('Fitting SE with EMA TL + Drude')
                    opt_param = least_squares(self.TLD_SE_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting SE TL + Drude')

                    opt_param = least_squares(self.TLD_SE,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
            if Model == 'Tauc Lorentz':
                if EMA == 'true':
                    print('Fitting SE with EMA TL')
                    while self._is_running == 'True':
                        opt_param = least_squares(self.TL_SE_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                        if self._is_running == 'False':
                            break
                elif EMA == 'false':
                    print('Fitting SE TL')

                    opt_param = least_squares(self.TL_SE,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol)
            if Model == 'Cauchy':
                if EMA == 'true':
                    print('Fitting SE with EMA Cauchy')
                    opt_param = least_squares(self.C_SE_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting SE Cauchy')

                    opt_param = least_squares(self.C_SE,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol)
            if Model == 'Sellmeier':
                if EMA == 'true':
                    print('Fitting SE with EMA Sellmeier')
                    opt_param = least_squares(self.SM_SE_rough,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol) 
                elif EMA == 'false':
                    print('Fitting SE Sellmeier')

                    opt_param = least_squares(self.SM_SE,
                                              initial_guess, args=args, 
                                              verbose=2, method = 'lm'
                                              , ftol = ftol, xtol = xtol, gtol =gtol)

        print('fitting complete')
        print(opt_param)
            
        self.results = {'Optimization': opt_param, 'params': opt_param.x}
        return self.results    

    
class OptimizerWorker(QObject):
    finished = pyqtSignal()
    result_ready = pyqtSignal(dict)
    iteration_update = pyqtSignal(int, list, float, list, list, list)
    iteration_update_all = pyqtSignal(int, list, float,float, list, list, list, list, list)
    def __init__(self, model_inputs, RT_inputs, SE_inputs):
        super().__init__()
        
        self.model_inputs = model_inputs
        self.SE_inputs = SE_inputs
        self.RT_inputs = RT_inputs
        self.optimizer = Optimizer(model_inputs, SE_inputs, RT_inputs)
    def run(self):
        if self.model_inputs['RT fit'] and self.model_inputs['SE fit']:
            result = self.optimizer.model(self.iteration_update_all)
            self.result_ready.emit(result)
            self.finished.emit()

        else:
            result = self.optimizer.model(self.iteration_update)
            self.result_ready.emit(result)
            self.finished.emit()
    
        
class Calculation():
    def __init__(self, model_inputs, RT_inputs, SE_inputs, results):
        self.model_inputs = model_inputs
        self.RT_inputs = RT_inputs
        self.SE_inputs = SE_inputs
        self.results = results
    def calculation_RT(self, lam_vac):
        checkbox_RT_fit_checked = self.model_inputs.get('RT fit', None)

        subs_RT = self.RT_inputs['subs_RT']

        theta_0_R = self.RT_inputs['theta_0_R']
        theta_0_T = self.RT_inputs['theta_0_T']
        d_s_RT = self.RT_inputs['d_s_subs']
        DiE_model = self.model_inputs['Model']
        
        
        n = self.model_inputs[DiE_model]['n_osz']
        EMA = self.model_inputs[DiE_model]['EMA']
        
        subs_type = self.RT_inputs['c_subs']
        if subs_type == 'incoherent':
            if EMA == 'true':
                c_list = ['i', 'c', 'c', 'i','i']
            elif EMA == 'false':
                c_list = ['i', 'c', 'i','i']
        else:
            if EMA == 'true':
                c_list = ['i', 'c', 'c', 'c','i']
            elif EMA == 'false':
                c_list = ['i', 'c', 'c','i']

        if checkbox_RT_fit_checked:
            params = self.results['params']
        else:
            params = self.model_inputs[DiE_model]['params']
        args = {}
        args = ({'d_s_RT': d_s_RT, 
                    'theta_0_T': theta_0_T, 
                    'theta_0_R': theta_0_R, 
                    'c_list': c_list, 
                    'subs_RT' : subs_RT,
                    'Model': DiE_model,
                    'EMA': EMA,
                    'n': n})
                
    
        R, T, n, k = model(lam_vac, params, args)
        return R, T, n, k  
    def calculation_SE(self, lam_vac, theta):
        checkbox_SE_fit_checked = self.model_inputs.get('SE fit', None)
        subs_SE = self.SE_inputs['subs_SE']


        DiE_model = self.model_inputs['Model']
        n = self.model_inputs[DiE_model]['n_osz']
        EMA = self.model_inputs[DiE_model]['EMA']
        
        if checkbox_SE_fit_checked:
            params = self.results['params']
        else:
            params = self.model_inputs[DiE_model]['params']
            
        args = {}
        args = ({  'subs_SE': subs_SE,
                   'n': n,
                   'EMA': EMA,
                   'Model': DiE_model})
        
       
        psi, delta, n, k = model_2(lam_vac, params, args, theta)
        return psi, delta, n, k
    def calculation_nk(self, lam_vac):
        

        DiE_model = self.model_inputs['Model']
        n =  self.model_inputs[DiE_model]['n_osz']
        params_0 = self.model_inputs[DiE_model]['params']
            
        params = np.zeros(len(params_0)+1)
        params[0] = n
        params[1:] = params_0
        
        if DiE_model == 'Tauc Lorentz':
            epsilon_1, epsilon_2, n, k = TL_multi(lam_vac, params)
        
        if DiE_model == 'Tauc Lorentz + Drude':
            epsilon_1, epsilon_2, n, k = TL_drude(lam_vac, params)

        if DiE_model == 'Cauchy':
     
            epsilon_1, epsilon_2, n, k = cauchy(lam_vac, params)
        if DiE_model == 'Sellmeier':

            epsilon_1, epsilon_2, n, k = sellmeier(lam_vac, params)
            
        return epsilon_1, epsilon_2, n, k

    def error_RT(self,lam_examp, R, T, alpha_R, alpha_T):
                
        N = len(self.RT_inputs['lam_vac'])
        T_data = self.RT_inputs.get('T_raw', None)
        R_data = self.RT_inputs.get('R_raw', None)
        
        error_T = 0
        error_R = 0
        if T_data is not None:
            T_exp = interp1d(self.RT_inputs['lam_vac_raw'], T_data, kind='linear')(lam_examp)
            error_T = alpha_T*(T_exp - T)**2
        if R_data is not None:
            R_exp = interp1d(self.RT_inputs['lam_vac_raw'], R_data, kind='linear')(lam_examp) 
            error_R = alpha_R*(R_exp - R)**2
        
        Error_tot = np.sqrt(1 / N * (np.sum(error_T) + np.sum(error_R)))*100
        return Error_tot
    def error_SE(self, lam_examp, psis, deltas, alpha_psi, alpha_delta):
      
        theta_values= self.SE_inputs.get('thetas', None)
        
        N = len(self.SE_inputs['lam_vac'])*len(theta_values)
        psi_exp = {}
        delta_exp = {}
        error_SE = []
        psis_data = self.SE_inputs.get('psis_raw', None)
        deltas_data = self.SE_inputs.get('deltas_raw', None)
        Error_tot = 0
        if psis_data is not None and deltas_data is not None:

            for i, theta in enumerate(theta_values):
                psi_exp[f'{theta}'] = interp1d(self.SE_inputs['lam_vac_raw'], psis_data[i], kind='linear')(lam_examp)
                delta_exp[f'{theta}'] = interp1d(self.SE_inputs['lam_vac_raw'], deltas_data[i], kind='linear')(lam_examp)
                error_psi = alpha_psi*(psi_exp[f'{theta}'] - psis[:,i])**2
                error_delta = alpha_delta*(delta_exp[f'{theta}'] - deltas[:,i])**2
                error_SE.extend(error_psi)
                error_SE.extend(error_delta)

                Error_tot = np.sqrt(1 / N * np.sum(error_SE)) * 180 / np.pi

        return Error_tot
    def error_tot(self, lam_examp, R, T, psis, deltas, alpha_psi, alpha_delta, alpha_R, alpha_T):
        N = len(self.RT_inputs['lam_vac'])
        T_data = self.RT_inputs.get('T_raw', None)
        R_data = self.RT_inputs.get('R_raw', None)
        
        error_T = 0
        error_R = 0
        if T_data is not None:
            T_exp = interp1d(self.RT_inputs['lam_vac_raw'], T_data, kind='linear')(lam_examp)
            error_T = alpha_T*(T_exp - T)**2
        if R_data is not None:
            R_exp = interp1d(self.RT_inputs['lam_vac_raw'], R_data, kind='linear')(lam_examp) 
            error_R = alpha_R*(R_exp - R)**2
        
        Error_RT = np.sqrt(1 / N * (np.sum(error_R) + np.sum(error_T)))*100
        
        theta_values= self.SE_inputs.get('thetas', None)
        if theta_values == None:
            theta_values = [50,60,70]
        N = len(self.SE_inputs['lam_vac'])*len(theta_values)
        psi_exp = {}
        delta_exp = {}
        error_SE = []
        psis_data = self.SE_inputs.get('psis_raw', None)
        deltas_data = self.SE_inputs.get('deltas_raw', None)
        Error_SE = 0
        
        if psis_data is not None and deltas_data is not None:
            for i, theta in enumerate(theta_values):
                psi_exp[f'{theta}'] = interp1d(self.SE_inputs['lam_vac_raw'], psis_data[i], kind='linear')(lam_examp)
                delta_exp[f'{theta}'] = interp1d(self.SE_inputs['lam_vac_raw'], deltas_data[i], kind='linear')(lam_examp)
                error_psi = alpha_psi*(psi_exp[f'{theta}'] - psis[:, i])**2
                error_delta = alpha_delta*(delta_exp[f'{theta}'] - deltas[:, i])**2
                error_SE.extend(error_psi)
                error_SE.extend(error_delta)

                Error_SE = np.sqrt(1 / N * np.sum(error_SE)) * 180 / np.pi
        
        return Error_RT, Error_SE
  
    def calculation(self):
        print('Calculation:')
        checkbox_RT_checked = self.model_inputs.get('RT calc', None)
        checkbox_SE_checked = self.model_inputs.get('SE calc', None)
        checkbox_RT_fit_checked = self.model_inputs.get('RT fit', None)
        checkbox_SE_fit_checked = self.model_inputs.get('SE fit', None)
        
        alpha_R = self.model_inputs.get('model_alpha_R', None)
        alpha_T = self.model_inputs.get('model_alpha_T', None)
        alpha_psi = self.model_inputs.get('model_alpha_psi', None)
        alpha_delta = self.model_inputs.get('model_alpha_delta', None)

   
        lam_min_plot = self.model_inputs['lam_min'] 
        lam_max_plot = self.model_inputs['lam_max'] 
        

        theta_values = self.SE_inputs.get('thetas', None)
        
    
        if (checkbox_RT_checked or checkbox_RT_fit_checked) and (checkbox_SE_checked or checkbox_SE_fit_checked):
            lam_RT = self.RT_inputs['lam_vac_reg']
            lam_SE = self.SE_inputs['lam_vac_reg']
            if len(lam_RT) == 0 and len(lam_SE) == 0:
                lam = np.linspace(lam_min_plot, lam_max_plot, 100)
                self.model_inputs['lam_examp']= lam
                lam_RT = lam
                lam_SE = lam
            R_tmm_opt = []
            T_tmm_opt = []
            n_opt = []
            k_opt = []
            
            print('start RT/SE based nk-calculation')
            R_tmm_opt, T_tmm_opt, n_opt, k_opt = np.transpose(np.array([self.calculation_RT(lam_vac) for lam_vac in lam_RT]))
            
            num_theta = len(theta_values)
            
            # Arrays für Psi- und Delta-Optimalwerte
            psi_opt = np.zeros((len(lam_SE), num_theta))
            delta_opt = np.zeros((len(lam_SE), num_theta))

            for i, theta in enumerate(theta_values):
                psi_opt[:, i], delta_opt[:, i], _, _ = np.transpose(np.array([self.calculation_SE(lam_vac, theta) 
                                                                         for lam_vac in lam_SE]))
            

            
            Error_RT = self.error_RT(lam_RT, R_tmm_opt, T_tmm_opt, alpha_R, alpha_T)
            Error_SE = self.error_SE(lam_SE, psi_opt, delta_opt, alpha_psi, alpha_delta)
            
            self.results['R'] = R_tmm_opt
            self.results['T'] = T_tmm_opt
            self.results['psi'] = psi_opt
            self.results['delta'] = delta_opt
            self.results['Error_RT'] = Error_RT
            self.results['Error_SE'] = Error_SE
            self.results['n'] = n_opt
            self.results['k'] = k_opt
            
        elif checkbox_RT_checked or checkbox_RT_fit_checked:
            lam_examp = self.RT_inputs['lam_vac']

            if len(lam_examp) == 0:
                lam_examp = np.linspace(lam_min_plot, lam_max_plot, 100)
                self.model_inputs['lam_examp']= lam_examp
            R_tmm_opt = []
            T_tmm_opt = []
            n_opt = []
            k_opt = []
            
            print('start RT-based nk-calculation')

            R, T, n_opt, k_opt= np.transpose(np.array([self.calculation_RT(lam_vac) 
                                                          for lam_vac in lam_examp]))
            Error_RT = self.error_RT(lam_examp, R, T, alpha_R, alpha_T)
            self.results['R'] = R
            self.results['T'] = T
            
            self.results['Error_RT'] = Error_RT
            # self.results['epsilon_1'] = epsilon_1_opt
            # self.results['epsilon_2'] = epsilon_2_opt
            self.results['n'] = n_opt
            self.results['k'] = k_opt
        elif checkbox_SE_checked or checkbox_SE_fit_checked:
            num_theta = len(theta_values)
            
            lam_examp = self.SE_inputs['lam_vac']
            if len(lam_examp) == 0:
                lam_examp = np.linspace(lam_min_plot, lam_max_plot, 100)
                self.model_inputs['lam_examp']= lam_examp
            # Arrays für Psi- und Delta-Optimalwerte
            psi_opt = np.zeros((len(lam_examp), num_theta))
            delta_opt = np.zeros((len(lam_examp), num_theta))
            n_opt = []
            k_opt = []
            print('start SE-based nk-calculation ')

            for i, theta in enumerate(theta_values):
                psi_opt[:, i], delta_opt[:, i], n_opt, k_opt  = np.transpose(np.array([self.calculation_SE(lam_vac, theta) 
                                                                         for lam_vac in lam_examp]))
                
           

            Error_SE = self.error_SE(lam_examp, psi_opt, delta_opt, alpha_psi, alpha_delta)
        
           
            self.results['psi'] = psi_opt
            self.results['delta'] = delta_opt
            
            self.results['Error_SE'] = Error_SE
            self.results['n'] = n_opt
            self.results['k'] = k_opt
        else:
            print('nk calculation only')
            
            lam_examp = np.linspace(lam_min_plot, lam_max_plot, 100)

            n_opt = []
            k_opt = []
            
            epsilon_1_opt, epsilon_2_opt, n_opt, k_opt = np.transpose(np.array([self.calculation_nk(lam_vac) 
                                                                                for lam_vac in lam_examp]))
            self.results['epsilon_1'] = epsilon_1_opt
            self.results['epsilon_2'] = epsilon_2_opt
            self.results['n'] = n_opt
            self.results['k'] = k_opt
        return self.results

class Result(QtWidgets.QMainWindow, Ui_MainWindow_results):
    def __init__(self, model_inputs, SE_inputs, RT_inputs,results):
        super(Result, self).__init__()
        self.setupUi(self)

        # Variable für die FigureCanvas-Instanz
        self.model_inputs = model_inputs
        self.SE_inputs = SE_inputs
        self.RT_inputs = RT_inputs
        self.results = results
        
        self.lam_min_plot = self.model_inputs.get('lam_min', None)
        self.lam_max_plot = self.model_inputs.get('lam_max', None)
        self.result_table.setColumnCount(2)
        self.export_data.clicked.connect(self.save_data)
        #plot_layout
        
    def clear_layout(self, layout):
        layout.deleteLater()
    def add_plot_layout(self):
        self.scroll_area = QtWidgets.QScrollArea(self.tab_2)
        self.scroll_area.setGeometry(20, 20, 1200, 800)
        self.scroll_area.setObjectName("scroll_area")
        
        # Erstelle ein Widget, das als Inhaltsbereich der ScrollArea dient
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_content.setObjectName("scroll_content")
        
        self.plot_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.plot_layout.setContentsMargins(20, 10, 10, 40)
        self.plot_layout.setObjectName("plot_layout")
        
        self.scroll_content.setLayout(self.plot_layout)
        self.scroll_content.setMinimumSize(1180, 1000)  

        # Erstelle das vertikale Layout innerhalb des Inhaltsbereichs
        

        # Setze das Inhaltswidget der ScrollArea
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll_area.setWidgetResizable(True)

    def show_results(self):
        RT_calc = self.model_inputs['RT calc']
        RT_fit = self.model_inputs['RT fit']
        SE_calc = self.model_inputs['SE calc']
        SE_fit = self.model_inputs['SE fit']
        Model = self.model_inputs['Model']
        
        if RT_fit or SE_fit:
            labels = self.model_inputs[Model]['labels']
            params = self.results['params']
            
            combined_data = {label: param for label, param in zip(labels, params)}
        elif RT_calc or SE_calc:
            labels = self.model_inputs[Model]['labels']
            params = self.model_inputs[Model]['params']

            combined_data = {label: param for label, param in zip(labels, params)}
        else:
            labels = self.model_inputs[Model]['labels']
            params = self.model_inputs[Model]['params']
          
            combined_data = {label: param for label, param in zip(labels, params)}
        for param, value in combined_data.items():
            row_position = self.result_table.rowCount()
            self.result_table.insertRow(row_position)
        
            param_item = QtWidgets.QTableWidgetItem(param)
            value_item = QtWidgets.QTableWidgetItem(str(value))
            
            self.result_table.setItem(row_position, 0, param_item)
            self.result_table.setItem(row_position, 1, value_item)

    def plot_nk(self):
        print('plot nk')
        
        if hasattr(self, 'scroll_area'):

            self.clear_layout(self.scroll_area)
        self.add_plot_layout()
        
        self.nk = pg.PlotWidget()
        self.nk.setBackground("w")
        self.plot_layout.addWidget(self.nk)

        n_opt = self.results['n']
        k_opt = self.results['k']
        
        lam_examp = np.linspace(self.lam_min_plot, self.lam_max_plot, 100)

        self.nk.plot(lam_examp, n_opt, name='n', pen=pg.mkPen(color='red'))
        self.nk.plot(lam_examp, k_opt, name='k', pen=pg.mkPen(color='orange'))
        
        self.nk.setLabel('bottom', 'Wellenlänge [nm]', units='', color='black')

        self.nk.setLabel('left', 'n/k', units='', color='black')

        # self.nk.addLegend()
        legend = self.nk.addLegend()
        legend.setVisible(True)
        legend.anchor((0, 1), (0, 1)) 

    def plot_RT(self):
        
        if hasattr(self, 'scroll_area'):

            self.clear_layout(self.scroll_area)
        self.add_plot_layout()
        
        self.nk = pg.PlotWidget()
        self.RT = pg.PlotWidget()

        self.nk.setBackground("w")
        self.RT.setBackground("w")
    
        self.plot_layout.addWidget(self.nk)
        self.plot_layout.addWidget(self.RT)
        
        lam_examp = self.RT_inputs.get('lam_vac', None)
        if len(lam_examp) == 0:
            lam_examp = self.model_inputs.get('lam_examp', None)

        R_data = self.RT_inputs.get('R_raw', None)
        T_data = self.RT_inputs.get('T_raw', None)
        lam_vac_RT = self.RT_inputs.get('lam_vac_raw', None)
        
        R_tmm_opt = self.results['R']
        T_tmm_opt = self.results['T']
        ER_RT = self.results['Error_RT']
 
        
        n_opt = self.results['n']
        k_opt = self.results['k']
        print('C create RT display plot')

        self.RT.plot(lam_examp, T_tmm_opt * 100, pen='b', name='T')
        self.RT.plot(lam_examp, R_tmm_opt * 100, pen='k', name='R')
        if T_data is not None:
            self.RT.plot(lam_vac_RT, T_data * 100, pen=pg.mkPen(color='b', style=QtCore.Qt.DashLine), name='exp T', )
        if R_data is not None:
            self.RT.plot(lam_vac_RT, R_data * 100, pen=pg.mkPen(color='k', style=QtCore.Qt.DashLine), name='exp R')

        text_item  = pg.TextItem(text=f'ER={ER_RT:.2f}%', color=(0, 0, 0), anchor=(1, 1), border=None)
        self.RT.addItem(text_item)
        text_item.setPos(0.98, 0.98)
        
        self.nk.plot(lam_examp, n_opt, name='n', pen=pg.mkPen(color='red'))
        self.nk.plot(lam_examp, k_opt, name='k', pen=pg.mkPen(color='orange'))
        
        self.RT.setLabel('bottom', 'Wellenlänge [nm]', units='', color='black')
       
        self.nk.setLabel('bottom', 'Wellenlänge [nm]', units='', color='black')

        self.RT.setLabel('left', 'R/T [%]', units='', color='black')
        self.nk.setLabel('left', 'n/k', units='', color='black')

        self.nk.addLegend()
        self.RT.addLegend()
        
    def plot_SE(self):
        if hasattr(self, 'scroll_area'):

            self.clear_layout(self.scroll_area)
        self.add_plot_layout()
        
        self.nk = pg.PlotWidget()
        self.plot_psi = pg.PlotWidget()
        self.plot_delta = pg.PlotWidget()
        
        self.nk.setBackground("w")
        self.plot_psi.setBackground("w")
        self.plot_delta.setBackground("w")

        self.plot_layout.addWidget(self.nk)
        self.plot_layout.addWidget(self.plot_psi)
        self.plot_layout.addWidget(self.plot_delta)

        lam_examp = self.SE_inputs.get('lam_vac', None)
        if len(lam_examp) == 0:
            lam_examp = self.model_inputs.get('lam_examp', None)
            
        lam_vac_SE = self.SE_inputs.get('lam_vac_raw', None)
        psis_data = self.SE_inputs.get('psis_raw', None)
        deltas_data = self.SE_inputs.get('deltas_raw', None)
        theta_values = self.SE_inputs.get('thetas', None)
        
        psi_opt = self.results['psi']
        delta_opt = self.results['delta']
        
        n_opt = self.results['n']
        k_opt = self.results['k']
        
        ER_SE = self.results['Error_SE'] 

        colors = ['black', 'green', 'purple', 'red', 'blue', 'magenta']
        
 
        for i, theta in enumerate(theta_values):
            
            self.plot_psi.plot(lam_examp, psi_opt[:, i] * 180 /np.pi, pen=pg.mkPen(color=colors[i]), name=f'psi {theta}°')
            
            
            self.plot_delta.plot(lam_examp, delta_opt[:, i] * 180 /np.pi, pen=pg.mkPen(color=colors[i]), name=f'delta {theta}°')
        
        for i, theta in enumerate(theta_values):

            if psis_data is not None:
                for i, data in enumerate(psis_data):
                    self.plot_psi.plot(lam_vac_SE, data * 180 /np.pi, pen=pg.mkPen(color=colors[i], style=QtCore.Qt.DashLine),  name=f'exp psi ${theta_values[i]}°$')
            if deltas_data is not None:
                for i, data in enumerate(deltas_data):
                    self.plot_delta.plot(lam_vac_SE, data * 180 /np.pi, pen=pg.mkPen(color=colors[i], style=QtCore.Qt.DotLine), name=f'exp delta ${theta_values[i]}°$')
                
        
        text_item  = pg.TextItem(text=f'ER={ER_SE:.2f}%', color=(0, 0, 0), anchor=(1, 1), border=None)
        self.plot_psi.addItem(text_item)
        text_item.setPos(0.98, 0.98)
  
        self.nk.plot(lam_examp, n_opt, name='n', pen=pg.mkPen(color='red'))
        self.nk.plot(lam_examp, k_opt, name='k', pen=pg.mkPen(color='orange'))
        
        self.plot_delta.setLabel('bottom', 'Wellenlänge [nm]', units='', color='black')
        self.plot_psi.setLabel('bottom', 'Wellenlänge [nm]', units='', color='black')
        self.nk.setLabel('bottom', 'Wellenlänge [nm]', units='', color='black')

        self.plot_delta.setLabel('left', 'Delta [deg]', units='', color='black')
        self.plot_psi.setLabel('left', 'psi [deg]', units='', color='black')
        self.nk.setLabel('left', 'n/k', units='', color='black')

        
        self.nk.addLegend()
        self.plot_psi.addLegend()
        self.plot_delta.addLegend()
    def plot_RTSE(self):
        if hasattr(self, 'scroll_area'):

            self.clear_layout(self.scroll_area)
        self.add_plot_layout()
        
        self.nk = pg.PlotWidget()
        self.RT = pg.PlotWidget()
        self.plot_psi = pg.PlotWidget()
        self.plot_delta = pg.PlotWidget()
        
        self.nk.setBackground("w")
        self.RT.setBackground("w")
        self.plot_psi.setBackground("w")
        self.plot_delta.setBackground("w")

        self.plot_layout.addWidget(self.nk)
        self.plot_layout.addWidget(self.RT)
        self.plot_layout.addWidget(self.plot_psi)
        self.plot_layout.addWidget(self.plot_delta)

       
        lam_examp_RT = self.RT_inputs.get('lam_vac_reg', None)
        if len(lam_examp_RT) == 0:
            lam_examp_RT = self.model_inputs.get('lam_examp', None)
            
        R_data = self.RT_inputs.get('R_raw', None)
        T_data = self.RT_inputs.get('T_raw', None)
        lam_vac_RT = self.RT_inputs.get('lam_vac_raw', None)
        
        R_tmm_opt = self.results['R']
        T_tmm_opt = self.results['T']
        ER_RT = self.results['Error_RT']
        
       
        lam_examp_SE = self.SE_inputs.get('lam_vac_reg', None)
        if len(lam_examp_SE) == 0:
            lam_examp_SE = self.model_inputs.get('lam_examp', None)
        lam_vac_psidelta = self.SE_inputs.get('lam_vac_raw', None)
        psis_data = self.SE_inputs.get('psis_raw', None)
        deltas_data = self.SE_inputs.get('deltas_raw', None)
        theta_values = self.SE_inputs.get('thetas', None)
        
        psi_opt = self.results['psi']
        delta_opt = self.results['delta']
        
        n_opt = self.results['n']
        k_opt = self.results['k']
        
        ER_SE = self.results['Error_SE'] 
        

        self.RT.plot(lam_examp_RT, T_tmm_opt, pen='b', name='T')
        self.RT.plot(lam_examp_RT, R_tmm_opt, pen='k', name='R')
        if T_data is not None:
            self.RT.plot(lam_vac_RT, T_data, pen=pg.mkPen(color='b', style=QtCore.Qt.DashLine), name='exp T', )
        if R_data is not None:
            self.RT.plot(lam_vac_RT, R_data,pen=pg.mkPen(color='k', style=QtCore.Qt.DashLine), name='exp R')

        text_item  = pg.TextItem(text=f'ER={ER_RT:.2f}%', color=(0, 0, 0), anchor=(1, 1), border=None)
        self.RT.addItem(text_item)
        text_item.setPos(0.98, 0.98)

        colors = ['black', 'green', 'purple', 'red', 'blue', 'magenta']
        
 
        for i, theta in enumerate(theta_values):
            
            self.plot_psi.plot(lam_examp_SE, psi_opt[:, i], pen=pg.mkPen(color=colors[i]), name=f'psi {theta}°')
            
            
            self.plot_delta.plot(lam_examp_SE, delta_opt[:, i], pen=pg.mkPen(color=colors[i]), name=f'delta {theta}°')
        
        for i, theta in enumerate(theta_values):

            if psis_data is not None:
                for i, data in enumerate(psis_data):
                    self.plot_psi.plot(lam_vac_psidelta, data, pen=pg.mkPen(color=colors[i], style=QtCore.Qt.DashLine),  name=f'exp psi ${theta_values[i]}°$')
            if deltas_data is not None:
                for i, data in enumerate(deltas_data):
                    self.plot_delta.plot(lam_vac_psidelta, data, pen=pg.mkPen(color=colors[i], style=QtCore.Qt.DotLine), name=f'exp delta ${theta_values[i]}°$')
                
        
        
        text_item  = pg.TextItem(text=f'ER={ER_SE:.2f}%', color=(0, 0, 0), anchor=(1, 1), border=None)
        self.plot_psi.addItem(text_item)
        text_item.setPos(0.98, 0.98)
        
        lam_examp = lam_examp_SE


        self.nk.plot(lam_examp, n_opt, name='n', pen=pg.mkPen(color='red'))
        self.nk.plot(lam_examp, k_opt, name='k', pen=pg.mkPen(color='orange'))
        
        self.RT.setLabel('bottom', 'Wellenlänge [nm]', units='', color='black')
        self.plot_delta.setLabel('bottom', 'Wellenlänge [nm]', units='', color='black')
        self.plot_psi.setLabel('bottom', 'Wellenlänge [nm]', units='', color='black')
        self.nk.setLabel('bottom', 'Wellenlänge [nm]', units='', color='black')

        
        self.RT.setLabel('left', 'R/T [%]', units='', color='black')
        self.plot_delta.setLabel('left', '\\Delta [deg]', units='', color='black')
        self.plot_psi.setLabel('left', '\\psi [deg]', units='', color='black')
        self.nk.setLabel('left', 'n/k', units='', color='black')

        
        self.nk.addLegend()
        self.plot_psi.addLegend()
        self.plot_delta.addLegend()
        self.RT.addLegend()
    def save_data(self):
        sample_name = self.sample.text()
        export_data = [self.lam_examp]

        # Kopfzeile initialisieren
        header = ['Wavelength [nm]']
        self.theta_values = self.SE_inputs.get('thetas', None)

        # Daten für n und k
        if self.save_nk.isChecked():
            n = self.results.get('n', [])
            k = self.results.get('k', [])
            header.extend(['n', 'k'])
            export_data.extend([n, k])
    
        # Daten für R und T
        if self.save_RT.isChecked():
            R = self.results.get('R', [])
            T = self.results.get('T', [])
            header.extend(['R', 'T'])
            export_data.extend([R, T])
    
        # Daten für psi und delta
        if self.save_SE.isChecked():
            psis = self.results.get('psi', [])
            deltas = self.results.get('delta', [])
            for i,theta in enumerate(self.theta_values):
                header.extend([f'psi {theta}', f'delta {theta}'])
                psi = (psis.T)[i]
                delta = (deltas.T)[i]
                export_data.extend([psi, delta])
    
        # Transponieren der Daten
        export_data = list(zip(*export_data))

        # Tkinter root initialisieren und Dateidialog öffnen
        root = tk.Tk()
        root.withdraw()  # Das Hauptfenster ausblenden
    
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save as",
            initialfile="exported_data.csv"
        )
    
        # Datei speichern, falls ein Pfad ausgewählt wurde
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Kopfzeile mit Sample-Namen
                writer.writerow([f'Sample-Name: {sample_name}'])
                # Daten-Kopfzeile
                writer.writerow(header)
                # Daten schreiben
                writer.writerows(export_data)
    
            print("Data export succesfull.")
        
class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))
    def flush(self):
        pass  # Notwendig für sys.stdout.
class ConsoleWindow(QtWidgets.QMainWindow):

    update_RT_plot_signal = pyqtSignal(list, list, list)
    update_SE_plot_signal = pyqtSignal(list, list, list, list)
    update_text_signal = pyqtSignal(str)
    
    def __init__(self):
        
        super().__init__()
        self.setWindowTitle('Console Output')
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        self.layout = QtWidgets.QVBoxLayout(central_widget)

        self.textEdit = QtWidgets.QTextEdit(self)
        self.layout.addWidget(self.textEdit)

        self.plotWidget = pg.GraphicsLayoutWidget(self)
        self.plotWidget.setBackground('w') 
        self.layout.addWidget(self.plotWidget)

        self.plot_item = self.plotWidget.addPlot(row = 0, col=0)
        self.plot_item.setLabel('bottom', 'Wavelength [nm]')
        
        self.plot_item_psi = self.plotWidget.addPlot(row=1, col=0)
        self.plot_item_psi.setLabel('bottom', 'Wavelength [nm]')
        self.plot_item_psi.setLabel('left', 'Psi [deg]')

        # Plot for Delta
        self.plot_item_delta = self.plotWidget.addPlot(row=1, col=1)
        self.plot_item_delta.setLabel('bottom', 'Wavelength [nm]')
        self.plot_item_delta.setLabel('left', 'Delta [deg]')
        
        
        
        self.T_curve = self.plot_item.plot(pen='b', name='T')
        self.R_curve = self.plot_item.plot(pen='r', name='R')
        
        self.T_data_curve = self.plot_item.plot(pen = None,symbol='o', symbolBrush='b', symbolSize=5, name='T_data')
        self.R_data_curve = self.plot_item.plot(pen=None, symbol='o', symbolBrush='r', symbolSize=5, name='R_data')
        

        self.psi_curve = self.plot_item.plot(pen='b', name='T')
        self.delta_curve = self.plot_item.plot(pen='r', name='R')
        
        self.psi_curves = []
        self.delta_curves = []
        
        self.psi_data_curves = []
        self.delta_data_curves = []

        self.stdout = EmittingStream(textWritten=self.append_text)
        sys.stdout = self.stdout


        self.update_text_signal.connect(self.append_text)
        
        self.update_RT_plot_signal.connect(self.display_RT_data)
        self.update_SE_plot_signal.connect(self.display_SE_data)

        self.stopButton = QtWidgets.QPushButton('Stop Calculation', self)
        self.layout.addWidget(self.stopButton)
        
    
    def append_text(self, text):
        self.textEdit.moveCursor(QtGui.QTextCursor.End)
        self.textEdit.insertPlainText(text)
        self.textEdit.moveCursor(QtGui.QTextCursor.End)

        # Scroll to the bottom of the textEdit to show the latest output
        self.textEdit.verticalScrollBar().setValue(self.textEdit.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.hide()
        event.ignore()
    def display_RT_data(self,lam_vac_RT, R, T):

        print('Show RT data')
        self.plot_item.setLabel('left', 'R/T [%]')

        self.T_data_curve.setData(lam_vac_RT, [t * 100 for t in T])

        self.R_data_curve.setData(lam_vac_RT, [r * 100 for r in R])
        self.plot_item.show()
    def display_SE_data(self,theta_values, lam_vac_SE, psis, deltas):
        print('Show SE data')

        
        # Clear old data curves
        for curve in self.psi_data_curves:
            self.plot_item_psi.removeItem(curve)
        for curve in self.delta_data_curves:
            self.plot_item_delta.removeItem(curve)

        # Clear lists
        self.psi_data_curves.clear()
        self.delta_data_curves.clear()

        # Add new data curves
        for i, theta in enumerate(theta_values):
            
            psi_curve = self.plot_item_psi.plot(pen='b', name=f'psi_{i}')
            delta_curve = self.plot_item_delta.plot(pen='r', name=f'delta_{i}')
            psi_data_curve = self.plot_item_psi.plot(pen=None, symbol='o', symbolBrush='b', symbolSize=5, name=f'psi_data_{theta}')
            delta_data_curve = self.plot_item_delta.plot(pen=None, symbol='o', symbolBrush='r', symbolSize=5, name=f'delta_data_{theta}')
            
            psi_data_curve.setData(lam_vac_SE, [psi * 180 / np.pi for psi in psis[i]])
            delta_data_curve.setData(lam_vac_SE, [delta * 180 / np.pi for delta in deltas[i]])
            
            self.psi_data_curves.append(psi_data_curve)
            self.delta_data_curves.append(delta_data_curve)
            self.psi_curves.append(psi_curve)
            self.delta_curves.append(delta_curve)


        self.plot_item_psi.show()
        self.plot_item_delta.show()

    def display_RT_plot(self, iteration, params, mse, lam_vac_RT, R, T):
        print('update Plot')
        
        # self.plot_item.clear()
        self.plot_item.setTitle(f"Iteration round {iteration}")
        self.T_curve.setData(lam_vac_RT, [t * 100 for t in T])  # Multiplikation mit 100 für Prozentwerte
        self.R_curve.setData(lam_vac_RT, [r * 100 for r in R])
      
        formatted_params = ', '.join(params)

        text = f'Iteration: {iteration}, [{formatted_params}], MSE: {mse:.2f}%\n'      
        # print(text)
        self.update_text_signal.emit(text)
    def display_SE_plot(self, iteration, params, mse, lam_vac_SE, psis, deltas):
        self.plot_item_psi.setTitle(f"Iteration round {iteration}")
        self.plot_item_delta.setTitle(f"Iteration round {iteration}")

       
        # Clear lists
        self.psi_data_curves.clear()
        self.delta_data_curves.clear()

        # Add new data curves
        for i, psi_curve in enumerate(self.psi_curves):
            
            psi_curve.setData(lam_vac_SE, [psi * 180 / np.pi for psi in psis[i]])

            
        for i, delta_curve in enumerate(self.delta_curves):
    
            delta_curve.setData(lam_vac_SE, [delta * 180 / np.pi for delta in deltas[i]])
            

        self.plot_item_psi.show()
        self.plot_item_delta.show()
        
        formatted_params = ', '.join(params)

        text = f'Iteration: {iteration}, [{formatted_params}], MSE: {mse:.2f}%\n'      
        # print(text)
        self.update_text_signal.emit(text)
    def display_RTSE_plot(self, iteration, params, mse_RT, mse_SE, lam_vac_all, psis, deltas, R, T):
        self.plot_item.setTitle(f"Iteration round {iteration}")
        self.T_curve.setData(lam_vac_all, [t * 100 for t in T])  # Multiplikation mit 100 für Prozentwerte
        self.R_curve.setData(lam_vac_all, [r * 100 for r in R])
      
        self.plot_item_psi.setTitle(f"Iteration round {iteration}")
        self.plot_item_delta.setTitle(f"Iteration round {iteration}")

       
        # Clear lists
        self.psi_data_curves.clear()
        self.delta_data_curves.clear()

        # Add new data curves
        for i, psi_curve in enumerate(self.psi_curves):
            
            psi_curve.setData(lam_vac_all, [psi * 180 / np.pi for psi in psis[i]])

            
        for i, delta_curve in enumerate(self.delta_curves):
    
            delta_curve.setData(lam_vac_all, [delta * 180 / np.pi for delta in deltas[i]])
            
        self.plot_item.show()

        self.plot_item_psi.show()
        self.plot_item_delta.show()
        
        formatted_params = ', '.join(params)

        text = f'Iteration: {iteration}, [{formatted_params}], MSE_RT: {mse_RT:.2f}%, MSE_SE: {mse_SE:.2f}%\n'      
        # print(text)
        self.update_text_signal.emit(text)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow_settings = Settings()
    console_window = ConsoleWindow()
    console_window.show()
    # Add the following line to make sure the window is shown before the event loop starts
    
    try:
        MainWindow_settings.show()
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print('Main stopped')

        MainWindow_settings.stop_signal.emit()
        sys.exit()

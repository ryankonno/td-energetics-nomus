
'''
This code designed to match an ideal force trace through optimisation of the stimulation frequency used to drive the model. 

Compute the force generated through this process 

Ryan Konno
r.konno@uq.edu.au
The University of Queensland
'''


# Import 
import numpy as np 
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import lib.plot_style
palette = ("#32cd9c", "#f67410", "#2b21b8", "#C21599", "#83d921", "#1ab6e9")
import sys 
sys.path.append('./')

from Models.ActivationModel import ActivationModel
from Models.MechanicsModel import MechModel 
from Models.InitialEnergeticsModel import EnergeticsModel
from Models.BioenergeticsModel import Bioenergetics

# Define parameters
# note: could also import parameters from parameters_muscle.py
params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 50, # s, end time of the stimulation
    'stim_freq': 150, # Hz, Freqeucy of the stimulation 
    'contr_freq': 1, # Hz, Freqeucy of the repeated contractions 
    'N_cycles': 5, # Number of repeated contractions
    'duty_factor': 0.5, # Fraction of the cycle to apply the stimulation

    # Define muscle fibre-type properties 
    'alpha_s': 0.25, # Fraction of slow-type fibres

    # Define the muscle for the simulation 
    'muscle': 'SLOW', # 'SOL' or 'EDL'

    # General muscle parameters
    'rho0':  1e6, # g/m^3, Density of muscle

        # Mouse data 
        'SLOW': {

            # Excitation-activation parameters
            # 'Tau_1': 0.038,  # s, Close et al., 1967
            # 'Tau_2': 0.22,  # s, Close et al., 1967
            # Mouse data
            'Tau_1': 0.038,  # s, Close et al., 1967
            'Tau_2': 0.051 * 2,  # s, Close et al., 1967
            # 'Tau_2': 0.051 * 3,  # s, Close et al., 1967
            # "K": 0.25, # mM, Concentration of ATP at 50% activation, TOO HIGH 
            "K": 0.1025, # mM, Concentration of ATP at 50% activation
            # "n": 3, # Hill coefficient for act mdoel
            "n": 1.99, # Hill coefficient for act mdoel

            'beta_ca_damp': 5, # Damping coefficients
            'freq_cutoff': 7, # hz, Cutoff frequency

            # Mechanical parameters 
            'F_0': 0, # N, Barclay and Weber 2004
            'l_0': 11e-3, # m, Barclay and Weber 2004
            'mass': 4.1e-3, # g, Barclay and Weber 2004
            'max_iso_stress': 2.37e5, # N/m^2, Barclay 1996
            'dedt_ce_max': 4,  # s^-1, Barclay 2010
            'kappa': 0.18,
            
            # Initial energetics model 
            'r_cxb': 0.3895461, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.130118, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'cxb_scale':  0.9815, # unitless, cxb scale factor
            'cxb_scale':  1.08897, # unitless, cxb scale factor
            'r_sl': 0.23808, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Bioenergetics parameters 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
            # New corrected values 
            'V_max_oxphos': 2 * 2.0255, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.742, # unitless, # original
            'r_rec': 0.7142 * 0.5 * 0.165e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest    

            # Other model parameters
            # Konno et al., 2025 model parameters 
            'r1': 0.6177,
            'r2': 0.2342,

        }, 
        'FAST': {

            # Excitation-activation parameters
            # 'Tau_1': 0.011,  # requested
            # 'Tau_2': 0.044,  # BH 2003, fibre bundle data
            # Mouse data
            'Tau_1': 0.011,  # requested
            'Tau_2': 0.011 * 2,  # BH 2003, fibre bundle data
            # 'Tau_2': 0.011 * 3,  # BH 2003, fibre bundle data
            # "K": 0.45, #  TOO HIGH 
            "K": 0.1025, # mM, Concentration of ATP at 50% activation
            # "n": 3, # Hill coefficient for activation model
            "n": 2.89, # Hill coefficient for activation model

            'beta_ca_damp': 5, # Damping coefficients
            'freq_cutoff': 7, # hz, Cutoff frequency

            # Mechanical parameters
            'F_0': 0, # N,
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g,
            'max_iso_stress': 3.01e5, # N/m^2, B1996
            'dedt_ce_max': 10, # s^-1, Barclay 2010
            'kappa': 0.25,

            # Initial energetics model
            'r_cxb':    1.67170557, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.199734, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'cxb_scale': 1.74863, # unitless, cxb scale factor
            'r_sl':  0.6994197, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Bioenergetics parameters
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992
            # New corrected values 
            'V_max_oxphos': 2 * 2.0255, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.742, # unitless, # original
            'r_rec': 0.7142 * 0.5 * 0.165e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest    

            # Other model parameters
            # Konno et al., 2025 model parameters
            'r1': 2.7919,
            'r2': 0.697,
        },

        # Other bioenergetics parameters
        # Assume constant across all species and muscle fibre-types
        'V_ck_f': 100,# 100, # mM/s, Kushmerick 1998
        'K_b': 1.11, #mM, MacFarland 1994
        'K_ia': 0.135, # mM, MacFarland 1994
        'K_eq': 1.77e2, # Assuming a pH of 7, Lawson 1979
        'K_iq': 3.5, # mM, MacFarland 1994
        'K_ib': 3.9, # mM, MacFarland 1994
        'K_p': 3.8, # mM, MacFarland 1994
        'Gatp': 60e3, # J/mol, Free energy of ATP (Barclay 2019)

        # Mechanical model SEE stiffness (unused)
        'k_see': 0, 
}


# Compute additional parameters
for muscle in ('SLOW', 'FAST'):
    # Maximum isometric force
    params[muscle]['F_0'] = (
        params[muscle]['mass'] / params['rho0'] /
        params[muscle]['l_0'] *
        params[muscle]['max_iso_stress']
    )


'''
This is the definitions of new functions for the model... 
these will be incoporated in later to the actual codes.
'''

def excToStimulus(t_vec, u_vec): 
    '''
    Takes as input an excitation u and outputs an array of discharge times. 

    This transfer function takes values from the space u\in[0,1] to stim_times\in[0,200]. 200 is set as the maximum stimulus frequency.
    '''

    # Scale the stimulus frequency based on excitation level u
    max_stim_freq = 200  # Hz, maximum stimulus frequency
    stim_freq_scaled = u_vec * max_stim_freq

    # Compute the period (handle zero division)
    period = np.zeros_like(stim_freq_scaled, dtype=float)
    nonzero_mask = stim_freq_scaled > 0
    period[nonzero_mask] = 1.0 / stim_freq_scaled[nonzero_mask]

    # Compute the stimulation times 
    stim_times = np.zeros_like(t_vec, dtype=int)

    # Get the firing times
    t_fire_vec = []
    
    # Find first time when u > 0
    t_first_idx_array = np.where(u_vec > 0)[0]    
    t_first_idx = int(t_first_idx_array[0])
    # Initialize with the first firing time
    t_fire_prev = t_vec[t_first_idx]
    t_fire_vec.append(t_fire_prev)
    
    # Iterate through remaining times and add spikes when period has elapsed
    for idx_curr in range(t_first_idx + 1, len(t_vec)):
        t_val = t_vec[idx_curr]
        if period[idx_curr] > 0 and t_val >= t_fire_prev + period[idx_curr] - 1e-12:
            t_fire_vec.append(t_val)
            t_fire_prev = t_val
    
    t_fire_vec = np.array(t_fire_vec)

    # Map each spike time to the nearest index in t and mark stim_times
    if t_fire_vec.size > 0:
        t_arr = np.asarray(t_vec)
        for st in t_fire_vec:
            idx = int(np.argmin(np.abs(t_arr - st)))
            stim_times[idx] = 1
            
    return stim_times

def getExcComponents(u_vec, alpha_s): 
    u_slow = alpha_s * np.sin(0.5 * np.pi * u_vec)
    u_fast = (1 - alpha_s) * (1 - np.cos(0.5 * np.pi * u_vec))
    return u_slow, u_fast 

def computeFibreRat(self, u):
        # Compute relative fibre-type fraction with respect to excitation
        u_slow = self.alpha_s * np.sin(0.5 * np.pi * u)
        u_fast = (1 - self.alpha_s) * (1 - np.cos(0.5 * np.pi * u))
        denom = u_slow + u_fast

        r = np.empty_like(denom, dtype=float)
        zero_mask = np.isclose(denom, 0.0)
        r[zero_mask] = self.alpha_s
        r[~zero_mask] = u_slow[~zero_mask] / denom[~zero_mask]
        return r

'''
This first part of the code simulates a feed-forward condition where we supply the input excitation and get the resultant force 
'''

# Define the input exctitation 
Nt = 1000
t_start = 0 
t_end = 2
t_vec = np.linspace(t_start,t_end,Nt)
u_max = 1
c_freq = 1 # Contraction frequency 
u_vec = u_max * 0.5 * (np.sin(np.pi * c_freq * t_vec - np.pi/2) + 1)

# Define the muscle fibre-type composition 
alpha_s = params['alpha_s']

# Now compute the stimulus times for the different fibre-types 
u_slow_vec, u_fast_vec = getExcComponents(u_vec, alpha_s)
stim_slow = excToStimulus(t_vec, u_slow_vec)
stim_fast = excToStimulus(t_vec, u_fast_vec)

# Convert stim times into the index values, ie the index where the vector is nonzero
stim_slow_idx = np.nonzero(stim_slow)[0]
stim_fast_idx = np.nonzero(stim_fast)[0]

# Now compute the force
act_model_slow = ActivationModel(params['SLOW'], t_vec) 
act_model_fast = ActivationModel(params['FAST'], t_vec) 

# Solve the model 
stim_slow_vec, ca_slow_vec, catn_slow_vec, act_slow_vec = act_model_slow.runExcAct(stim_slow_idx)
stim_slow_vec_nodamp, ca_slow_vec_nodamp, catn_slow_vec_nodamp, act_slow_vec_nodamp = act_model_slow.runExcAct(stim_slow_idx, damping=False)
stim_fast_vec, ca_fast_vec, catn_fast_vec, act_fast_vec = act_model_fast.runExcAct(stim_fast_idx)
stim_fast_vec_nodamp , ca_fast_vec_nodamp , catn_fast_vec_nodamp, act_fast_vec_nodamp  = act_model_fast.runExcAct(stim_fast_idx,damping=False)

# Combine to get total muscle activation (with damping)
act_muscle = alpha_s * act_slow_vec + (1 - alpha_s) * act_fast_vec 

fig, axs = plt.subplots(1,2,layout = 'constrained') 
ax = axs[0]
ax.plot(t_vec, ca_slow_vec, label = 'Damping, beta = 1')
ax.plot(t_vec, act_slow_vec_nodamp, label = 'no damping')
ax.plot(t_vec, act_slow_vec, label = 'Damping, beta = 1')
ax.legend() 
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalised []')
ax = axs[1]
ax.plot(t_vec, ca_fast_vec, label = 'Damping, beta = 1')
ax.plot(t_vec, act_fast_vec_nodamp, label = 'no damping')
ax.plot(t_vec, act_fast_vec, label = 'Damping, beta = 1')
ax.legend() 
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalised []')

# Plot output
fig, ax = plt.subplots()
ax.plot(t_vec,  alpha_s * act_slow_vec, label = 'ST')
ax.plot(t_vec, (1 - alpha_s) * act_fast_vec, label = 'FT')
ax.plot(t_vec, act_muscle, label = 'ST + FT')
ax.set_ylabel('Activation')
ax2 = ax.twinx()
ax2.plot(t_vec, u_vec, label = 'Excitation')
ax2.set_ylabel('Excitation')
ax.set_xlabel('Time (s)')
ax.legend()
plt.show()











    





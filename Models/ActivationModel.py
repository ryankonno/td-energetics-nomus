'''
Excitation-activation model code.

Ryan Konno
r.konno@uq.edu.au
The University of Queensland 
'''
###################################################################
# Import
import numpy as np
from scipy.integrate import solve_ivp

###################################################################
# Implementation of activation model from Mayfield et al. 2022.
class ActivationModel():
    def __init__(self, params, t):
        '''
        Initialisation
        '''

        # Intialize the time constants
        self.tau_1 = params['Tau_1']
        self.tau_2 = params['Tau_2']

        # Store the parameters
        self.params = params

        # Store the time vector
        self.t = t
        self.dt = t[1] - t[0]

    ###################################################################
    def runExcAct(self, t_stim_idx, damping = True, w_0 = 0.0048):
        '''
        Function to run the excitation-activation dynamics

        Input: 
            t_stim_idx: index of stimulation times
            w_0: stimulation width (default w_0 = 0.0048)
        Output: 
            stim_vec: vector of stimulation times (size of self.t). 1 if muscle is stimulated 0 otherwise
            ca_vec: Normalised free Ca concentrations
            catn_vec: Normalised CaTn concentrations
        '''

        # Now compute the Ca concentrations 
        ca_vec = self.computeCa(self.tau_1, self.tau_2, t_stim_idx, w_0 = w_0) 

        # Compute the bound Ca 
        if damping: 
            catn_vec = self.hillEquation(ca_vec)
            act = self.computeCaTn(ca_vec, self.dt) # with damping
        else:        
            catn_vec = self.hillEquation(ca_vec)
            act = catn_vec

        # Compute the stimulation values 
        stim_vec = self.stim(t_stim_idx, w_0 = w_0)

        return stim_vec, ca_vec, catn_vec, act

    ###################################################################
    def stim(self, t_stim_idx, w_0 = 0.0048):
        '''
        Determine the stimulus values. 
        
        This function converts the single instantaneous pulses into the times at which the Ca
        is released from the SR. This takes into account the attenuation of the pulse width 

        Input: 
            t_stim_idx: Index of stimulation times 
        Output: 
            stims: vector of stimulation times (size of self.t). 1 if muscle is stimulated 0 otherwise
        '''
        t_n_1 = None
        stims = np.zeros(np.size(self.t)) # Get the the stimulus
        dt = self.dt

        # Loop over stimulation indexes
        for stim_idx in t_stim_idx:
            t_n = self.t[stim_idx]
            # Compute the stimulus values for the action potential 
            A = 0.2 # Values from Mayfield et al. 2022
            r = 0.35 # s, from Mayfield et al. 2022
            if t_n_1 is None:
                # Use full base width for the first pulse (no prior ISI).
                width = w_0
            else:
                t_isi = t_n - t_n_1
                width = w_0 * (1 - A * np.exp(-t_isi/r)) # Compute the pulse width based on isi

            # Keep onset after t_n while ensuring at least one sample-wide pulse.
            width_eff = max(width, dt)
            stims = stims + (self.t > t_n) * (self.t <= t_n + width_eff)

            # Update previous firing time 
            t_n_1 = t_n

        return stims

    ###################################################################  
    def computeCa(self, tau_1_, tau_2_, MUAP_idx, w_0 = 0.0048):
        """
        Computes calcium levels using an analytical step-wise method.
        
        Inputs:
            tau_1_: Time constant for calcium activation (rise).
            tau_2_: Time constant for calcium decay (fall).
            MUAP_idx (array): Indices of stimulus events within self.t.
            w_0: Base stimulus pulse width (default 0.0048 s).
        
        Outputs:
            a: Array of ca levels
        """
        # Extract time vector
        t = self.t
        dt = t[1] - t[0]  # Time step size
        n = len(t)
        
        # Compute calcium dynamics using the analytical step-wise method
        stim_times = t[MUAP_idx]
        a_curr = 0
        a = np.zeros_like(t)
        prev_idx = 0
        next_idx = 1

        for idx, t_ in enumerate(t):
            # Update interstimulus interval
            if next_idx < len(stim_times) and t_ > stim_times[next_idx]:
                prev_idx += 1
                next_idx += 1

            t_isi = stim_times[next_idx] - stim_times[prev_idx] if next_idx < len(stim_times) else np.inf

            # Check if in stimulation region
            A = 0.2  # Values from Mayfield et al. 2022
            r = 0.35  # s, from Mayfield et al. 2022a
            width = w_0 * (1 - A * np.exp(-t_isi / r)) # Width decayse with successive pulses (mu firing)
            width_eff = max(width, dt)

            if (t_ <= stim_times[prev_idx] + width_eff) and (t_ > stim_times[prev_idx]):
                # If in stimulation region, update activation
                a_curr = 1 - (1 - a_curr) * np.exp(-dt / tau_1_)
            else:
                # If not in stimulation region, decay calcium
                a_curr = a_curr * np.exp(-dt / tau_2_)

            # Store the calcium level
            a[idx] = a_curr

        return a

    ###################################################################
    def hillEquation(self,a):
        '''
        Hill equation used for computing the bound ca 

        Input: 
            a: concentration of free ca 
        Output: 
            catn_vec_: Bound CaTn concentrations
        '''
        K = self.params["K"]
        n = self.params["n"]
        catn_vec_ = np.power(a,n) / (np.power(a,n) + K**n)
        return catn_vec_
    
    ###################################################################
    def computeCaTn(self, a, dt): 
        '''
        Compute the bound CaTn concentrations accounting for damping 
        Use a second order differential equation to smooth catn values.
        
        System: d²catn/dt² = (catn_raw - catn) / beta_ca_damp
        where:
          y[0] = catn (damped/smoothed concentration)
          y[1] = d(catn)/dt (velocity)
        
        Input: 
            a: concentration of free ca 
            dt: time step
        Output: 
            catn_vec_: Damped CaTn Concentrations 
        '''
        # Get the damping constant
        beta_ca_damp = self.params['beta_ca_damp']

        # get the cutoff frequency constant 
        omega_n = self.params['freq_cutoff'] * 2 * np.pi # radians / s

        def rhs(t, y): 
            # Get the target undamped catn from hill equation
            catn_raw = self.hillEquation(np.interp(t, self.t, a)) # This is used as input to determine the activation
            act = y[0] 
            dactdt = y[1]

            # New equation with a cutoff frequency 
            # y[0] = act # output (damped signal ) 
            # y[1] = dact/dt
            d2act_dt2 = omega_n**2 * (catn_raw - act) - 2 * beta_ca_damp * omega_n * dactdt
            
            return [y[1], d2act_dt2]

        # Solve the IVP 
        y0 = [0, 0]
        sol = solve_ivp(rhs, (self.t[0], self.t[-1]), y0, t_eval=self.t, method='RK45')
        act_ = sol.y[0, :]
        dactdt_ = sol.y[1, :]

        return act_
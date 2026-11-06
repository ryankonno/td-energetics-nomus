'''
Mechanical model

This model is implemented to compute the activation levels of a muscle given a muscle force as input.
Intrinsic properties of the model are based on Dick et al. 2017, and parameters are modified for the
given experiment.

Simplified version for single musclels (no solving  with SEE)

Author: Ryan Konno, University of Queensland
		r.konno@uq.edu.au
'''
################################################################################
# Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class MechModel():
    def __init__(self, l_0, l_t0, k_see, mass):
        '''
        Take a parameter set as input to the model
        '''

        # Import commonly used parameters throughout the model 

        self.l_0 = l_0                    # Optimal fibre length
        self.l_t0 = l_t0                    # Optimal fibre length

        self.k_see = k_see             # SEE component stiffness coefficient (linear constant or nonlinear parameter)

        self.mass = mass * 1e-3 # Mass of the muscle, assumes input in grams

        self.beta = 0.5  *  2 * np.sqrt(k_see * self.mass) # Compute the stiffness coefficient
        # print(f'beta = {self.beta}')
        # print(f'mass = {self.mass}')
        # print(f'ksee = {self.k_see}')

    ###############################################################
    #       INTRINSIC MUSCLE PROPERTIES
    # Define force-length and force-velocity properties for the muscle
    # From Dick et al. 2017
    def F_la(self, e_m):
        skew = 0.6
        round = 2.3
        width = 0.3  
        e_m_safe = np.clip(e_m, 1e-6, None) # Clip e_m to avoid negative values and numerical issues
        result = np.exp(-np.abs((e_m_safe**skew -1)/width)**round)
        return np.clip(result, 0, 1) # Ensure 0 outside physiological range

    def F_va(self, dedt_ce):
        return (1+dedt_ce/self.dedt_ce_max)/(1-dedt_ce/self.dedt_ce_max/self.kappa) * (dedt_ce < 0) \
                + (1.5 - 0.5*(1-dedt_ce/self.dedt_ce_max)/(1+7.56 * dedt_ce/ self.dedt_ce_max / self.kappa))*(dedt_ce >0)\
                + (dedt_ce == 0)
    
    def F_va_inverse(self, F):
        return self.dedt_ce_max * (self.kappa*(F-1)/(F+self.kappa) * (F > 1)\
            + (50*self.kappa*(F-1)/(25*self.kappa-378*F+567)) * (F<1)\
            + 0*(F==1))
    
    def F_lp(self, e_m): 
        return (2.64 * e_m**2 - 5.3 * e_m + 2.66) * (e_m > 1)

    
    # def k_see_fun(self, F, F_0 = 0, l_0 = 0):
    #     '''
    #     Function for the stiffness of SEE (mainly effect from tendon)
    #     '''
    #     # Constant stiffness
    #     stiffness = self.k_see * np.ones_like(F)

    #     # Exponential stiffness Lichtwark and Wilson 2008
    #     # Constants from Lichtwark and Wilson 2008
    #     # Q = 20 # Unitless
    #     # k_l = 325e3 # N/m
    #     # stiffness = k_l/F_0*l_0 * ( 1 + (0.9/-np.exp(Q * F)))

    #     return stiffness
    ##########################################################
    # Function to compute the muscle force 
    def computeForce(self, act_, e_m_, dedt_m_): 
        '''
        Function to compute the scaled muscle force give act, e_m, dedt_m
        '''
        return act_ * self.F_la(1 + e_m_) * self.F_va(dedt_m_) + self.F_lp(1 + e_m_)

    ##########################################################
    def solveMechModel(self, t_vec, act, dedt_ce_max_vec, kappa_vec): 
        
        # Save important variables 
        self.dedt_ce_max_vec = dedt_ce_max_vec
        self.kappa_vec = kappa_vec 

        # Add data to class 
        self.t_vec = t_vec 
        self.act = act

        # Define variables for now 

        # Solve the IVP
        sol = solve_ivp(self.rhs, (t_vec[0], t_vec[-1]), [0,0], t_eval=t_vec, max_step = 0.01)

        e_ce_vec = sol.y[0,] / self.l_0 
        dedt_ce_vec = sol.y[1,] / self.l_0 
        force = self.computeForce(self.act, e_ce_vec, dedt_ce_vec)


        return force, e_ce_vec, dedt_ce_vec

    def rhs(self, t, y): 
        '''
        y: [u,v] displacement and velocity of mass 
        '''
        # Extract variables 
        u = y[0,]
        v = y[1,]

        # Compute strain and strain rate for the muscle and tendon 
        eps_m = u / self.l_0 
        deps_m = v / self.l_0 
        eps_t = -u / self.l_t0
        deps_t = -v / self.l_t0 

        # Interpolate and compute muscle force
        act_curr = np.interp(t, self.t_vec, self.act)
        self.dedt_ce_max = np.interp(t, self.t_vec, self.dedt_ce_max_vec)
        self.kappa = np.interp(t, self.t_vec, self.kappa_vec)
        F_m = self.computeForce(act_curr, eps_m, deps_m)

        # Update damping from the current combined tangent stiffness
        # k_m = muscle_tangent_stiffness(t, eps_m, deps_m)
        # k_eq = combined_stiffness(k_m, k_t)
        # beta_dynamic = self.critical_damping(k_eq, mass) # Does not work
        F_t = self.k_see * eps_t + self.beta * deps_t 

        # Get the rhs
        rhs_1 = v.copy()
        rhs_2 = (F_t - F_m) / self.mass
        
        return (rhs_1, rhs_2)



    ######################################
    # # OLD VERSION
    # # Solve the model with accounting for the fibre-type properties of the muscle 
    # def solveModelMultipleMUs(self, t_vec, act, stiff_coeff, dedt_ce_max_vec, kappa_vec): 
    
    #     self.stiff_coeff = stiff_coeff
    #     self.dedt_ce_max_vec = dedt_ce_max_vec
    #     self.kappa_vec = kappa_vec 

    #     # Add data to class 
    #     self.t_vec = t_vec 
    #     self.act = act

    #     # Solve the IVP
    #     from scipy.integrate import solve_ivp
    #     sol = solve_ivp(self.rhsMultipleMUs, (t_vec[0], t_vec[-1]), [0], t_eval=t_vec, max_step = 0.01)

    #     force = sol.y[0,]
    #     e_ce_vec = (self.l_0 - (force / stiff_coeff)) / self.l_0

    #     act_vec = np.interp(sol.t, self.t_vec, self.act) + 0.001
    #     dedt_ce_max_vec = np.interp(sol.t, self.t_vec, self.dedt_ce_max_vec)
    #     kappa_vec = np.interp(sol.t, self.t_vec, self.kappa_vec)

    #     force_fv_vec = force / (act_vec * self.F_la(e_ce_vec))
    #     # dedt_ce_vec = dedt_ce_max_vec * (
    #     #     kappa_vec * (force_fv_vec - 1) / (force_fv_vec + kappa_vec) * (force_fv_vec > 1)
    #     #     + (50 * kappa_vec * (force_fv_vec - 1) / (25 * kappa_vec - 378 * force_fv_vec + 567)) * (force_fv_vec < 1)
    #     #     + 0 * (force_fv_vec == 1)
    #     # )
    #     dedt_ce_vec = np.diff(e_ce_vec, prepend=0) / np.diff(t_vec, prepend=0)

    #     return force, e_ce_vec, dedt_ce_vec

    # def rhsMultipleMUs(self, t, y): 
        
    #     # Get the current activation 
    #     act_curr = np.interp(t, self.t_vec, self.act) + 0.001

    #     # Recompute the force-velocity parameters 
    #     self.dedt_ce_max = np.interp(t, self.t_vec, self.dedt_ce_max_vec)
    #     self.kappa = np.interp(t, self.t_vec, self.kappa_vec)

    #     # Get the force from the force-velocity relationship 
    #     f_m = y[0,]
    #     self.e_ce = 1 - f_m / self.stiff_coeff # Compute based on isometric contraction

    #     force_fv = f_m / (act_curr * self.F_la(self.e_ce)) # original 
    #     # force_fv = f_m / (act_curr * self.F_la(self.e_ce)) - 5 * dedt_ce # with damper 

    #     # Get the current stretch rate
    #     dedt_ce = self.F_va_inverse(force_fv)

    #     return self.stiff_coeff * (- dedt_ce)
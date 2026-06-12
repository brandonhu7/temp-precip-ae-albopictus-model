# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 17:37:08 2025

@author: Brandon
"""

#temperature only model

#initially, i created this model as an epidemiological model for chikungunya, which is why some remnants of that project are still in the code
#i decided to pivot and focus on the effect of incorporating precipitation instead
#no infection was initiated (0 infected humans and 0 infected mosquitoes) so the epidemiological stuff does not affect the model


import numpy as np
from scipy.integrate import solve_ivp 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def time_to_temp(t,M,T_a,tau):
    return M - T_a*np.cos(2*np.pi/52*(t-tau))

def temp_briere(T, T_0, T_M, c):
    return np.maximum(0, c*T*(T-T_0)*np.sqrt(np.maximum(T_M-T,0)))

def temp_quadratic(T, T_0, T_M, c):
    return np.maximum(0, c*(T-T_0)*(T_M-T))

def temp_lambda(T):
    return (np.maximum(-.3379*np.power(T,2) + 16.86*T - 142.8,0))/(2*(np.maximum(0.045*np.power(T,2) - 2.717*T + 44.41,0)))

def model(y,t,k,temp):  
    # Mosquitoes
   
    EGG_V = y[0] #Number of eggs
    J_V = y[1] #Number of Juveniles
   
    S_V = y[2] #susceptible vectors (adult females)
    E1_V = y[3] #exposed stage 1 vectors
    E2_V = y[4] #exposed stage 2 vectors
    E3_V = y[5] #exposed stage 3 vectors
    I_V = y[6] #infected vectors
    N_V = S_V + E1_V + E2_V + E3_V + I_V #total number of vectors

   
    # Humans
   
    S_H = y[7] #sus humans
    E_H = y[8] #exp humans
    I_H = y[9] #inf humans
    R_H = y[10] #rec humans
   
    N_H = S_H + E_H + I_H + R_H #total number of humans
   

    C_H = y[11]
   
    # Constant parameters
   
    sigma_H = k[0] #rate at which humans go from exposed to infectious
    gamma_H = k[1] #rate at which humans go from infectious to recovered
    alpha = np.power(10,-k[5]) #density dependent juvenile mortality  
   
    beta = k[3] #also density dependent juvenile mortality  
    theta = k[4] #total number of exposed classes (stages 1 2 3)
   
    c_n = temp[3]
    T_0_n = temp[4]
    T_M_n = temp[5]
   
    c_mA = temp[6]
    T_0_mA = temp[7]
    T_M_mA = temp[8]
   
    c_a = temp[9]
    T_0_a = temp[10]
    T_M_a = temp[11]
   
    c_b = temp[12]
    T_0_b = temp[13]
    T_M_b = temp[14]
   
    c_c = temp[15]
    T_0_c = temp[16]
    T_M_c = temp[17]
   
    c_PDR = temp[18]
    T_0_PDR = temp[19]
    T_M_PDR = temp[20]
   
    c_h = temp[21]
    T_0_h = temp[22]
    T_M_h = temp[23]
   
    M = temp[30]
    T_a = temp[31]
    tau = temp[32]
    small_number = .0000001
    T = time_to_temp(t,M,T_a,tau)
   
    lambda0 = 7*temp_lambda(T) #eggs laid
    nu = 7*temp_briere(T, T_0_n, T_M_n, c_n) #j to a development rate
    mu_a = 7/((temp_quadratic(T, T_0_mA, T_M_mA, c_mA)+small_number)) #adult lifespan
    a = 7*temp_briere(T, T_0_a, T_M_a, c_a) #biting rate
    b = temp_briere(T, T_0_b, T_M_b, c_b) #probability that bitten human becomes infected
    c = temp_briere(T, T_0_c, T_M_c, c_c) #probability that mosquito biting infected human becomes infected (opposite of b)
    PDR = 7*temp_briere(T, T_0_PDR, T_M_PDR, c_PDR) #extrinsic incubation rate
    h = 7*temp_quadratic(T, T_0_h, T_M_h, c_h) #hatch rate
    mu_EGG = 7*0.212 #lifespan of eggs that dont hatch
    mu_J = 7*0.0816 #lifespan of juveniles that dont become adults
   
    beta_HV = a*c #mosquitoes to humans
    beta_VH = a*b #humans to mosquitoes
    rho = theta*PDR #transition rate between e1 e2 and e3 for vectors
   
   
   
   
    #human diff equations
   
    dS_Hdt = -((beta_VH*I_V)/N_H)*S_H
    dE_Hdt = ((beta_VH*I_V)/N_H)*S_H - sigma_H*E_H
    dI_Hdt = sigma_H*E_H - gamma_H*I_H
    dR_Hdt = gamma_H*I_H
    dC_Hdt = sigma_H*E_H
    #mosquito diff equations
   
    dEGG_Vdt = lambda0*N_V - h*EGG_V - mu_EGG*EGG_V
    dJ_Vdt = h*EGG_V - nu*J_V - mu_J*J_V - alpha*np.power(np.maximum(J_V, 0), beta)
    dS_Vdt = .5*nu*J_V-((beta_HV*I_H)/N_H + mu_a)*S_V
    dE1_Vdt = ((beta_HV*I_H)/N_H)*S_V - (mu_a + rho)*E1_V
    dE2_Vdt = rho*E1_V - (mu_a + rho)*E2_V
    dE3_Vdt = rho*E2_V - (mu_a + rho)*E3_V
    dI_Vdt = rho*E3_V - mu_a*I_V
   
    
    dydt = [dEGG_Vdt, dJ_Vdt, dS_Vdt, dE1_Vdt, dE2_Vdt, dE3_Vdt, dI_Vdt, dS_Hdt, dE_Hdt, dI_Hdt, dR_Hdt, dC_Hdt]
    return np.array(dydt, dtype=float)




# Initial Conditions
EGG_V = 125*99
J_V = 0

S_V = 0
E1_V = 0
E2_V = 0
E3_V = 0
I_V = 0
N_V = S_V + E1_V + E2_V + E3_V + I_V #total number of vectors

# Humans

S_H = 50
E_H = 0
I_H = 0
R_H = 0
C_H = 0

N_H = S_H + E_H + I_H + R_H #total number of humans

#init conds
y0 = [EGG_V, J_V, S_V, E1_V, E2_V, E3_V, I_V, S_H, E_H, I_H, R_H, C_H]


#constant parameters
sigma_H = 1.17 #rate at which humans go from exposed to infectious
gamma_H = 1.17 #rate at which humans go from infectious to recovered
x=7.5
alpha = np.power(10,-x) #density dependent juvenile mortality
beta = 4.4 #also density dependent juvenile mortality  
theta = 3 #total number of exposed classes (stages 1 2 3)

# store parameter values in a vector
p = [sigma_H, gamma_H, alpha, beta, theta, x]


#temp stuff

#temperature dependent stuff
c_l = 1 #placeholder, not used
T_0_l = 1 #placeholder, not used
T_M_l = 1 #placeholder, not used
   
c_n = 0.0000638
T_0_n = 8.60
T_M_n = 39.66
   
c_mA = 0.148
T_0_mA = 13.41
T_M_mA = 31.51
   
c_a =  0.000193
T_0_a = 10.25
T_M_a = 38.32
   
c_b = 0.000310
T_0_b = 15.84
T_M_b = 36.40
   
c_c = 0.000185
T_0_c = 3.62
T_M_c = 36.82
   
c_PDR = 0.000109
T_0_PDR = 10.39
T_M_PDR = 43.05
   
c_h = 0.000445
T_0_h = 12.99
T_M_h = 43.74
   
c_mEGG = 1 #placeholder, not used
T_0_mEGG = 1 #placeholder, not used
T_M_mEGG = 1 #placeholder, not used
   
c_mJ = 1 #placeholder, not used
T_0_mJ = 1 #placeholder, not used
T_M_mJ = 1 #placeholder, not used
   
delta=1 #fitted later

T_a = 2.78957
tau = 1.48823*52/(2*np.pi)
M = 24.5384

temp=[c_l, T_0_l, T_M_l, c_n, T_0_n, T_M_n, c_mA, T_0_mA, T_M_mA, c_a, T_0_a, T_M_a, c_b, T_0_b, T_M_b, c_c, T_0_c, T_M_c, c_PDR, T_0_PDR, T_M_PDR, c_h, T_0_h, T_M_h, c_mEGG, T_0_mEGG, T_M_mEGG, c_mJ, T_0_mJ, T_M_mJ, M, T_a, tau]

#mosquito data import
merged = pd.read_csv('wk1-141m.csv')

tB = 0 
tF = 141 
nTime = 141 

#the timespan
tspan = np.linspace(tB,tF,nTime)
tspansmall = np.linspace(0,40,40)
tspan_burnin = np.linspace(0,5*52,5*52) #start, end, number of timepoints



def objective_function(params_to_optimize):
    #replace parameters
    p_optimized = p.copy()
    p_optimized[5] = params_to_optimize[0]  #x

   
    wrapped_opt = lambda t, y: model(y, t, p_optimized, temp)

       
    sol_burnin_opt = solve_ivp(wrapped_opt, [tspan_burnin[0], tspan_burnin[-1]], y0, t_eval=tspan_burnin, method='BDF')
    init_conds_opt = sol_burnin_opt.y.T[-1, :]

    sol_opt = solve_ivp(wrapped_opt, [tspan[0], tspan[-1]], init_conds_opt, t_eval=tspan, method='BDF')
    ode_out = sol_opt.y.T


    #mosquito population (total vectors)
    EGG_V = ode_out[:, 0]
    model_eggs = (EGG_V) / 99.0  #div by 99 to match egg cohorting

    data_eggs = merged['individualCount'].values
    error = np.sum((model_eggs - data_eggs) ** 2)
    return error



x = 7.5
beta = 4.4
initial_guess = [x]


bounds = [(1, 10)]

result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)

optimized_x = result.x[0]

p[5] = optimized_x

#solveivp ode solver
#5 yrs
wrapped = lambda t, y: model(y, t, p, temp) #adding p and temp cuz solveivp usually only takes 2 var
sol_burnin = solve_ivp(wrapped, [tspan_burnin[0], tspan_burnin[-1]], y0, t_eval=tspan_burnin, method='BDF') #solveivp does its thing


ode_out_burnin = sol_burnin.y.T

# new simulations
init_conds = ode_out_burnin[(5*52)-1,:]

sol_optimized = solve_ivp(wrapped, [tspan[0], tspan[-1]], init_conds, t_eval=tspan)
ode_out = sol_optimized.y.T

#solution exraction
EGG_V = ode_out[:, 0]

a = EGG_V/99.0

#model output vs real data
plt.figure(figsize=(10, 6))
plt.scatter(merged['week'], merged['individualCount'], label='Data', color='indigo')
plt.plot(tspan, EGG_V/99.0, label='Model Prediction', color='dodgerblue', linewidth=3)
plt.xlabel('Time (weeks)')
plt.ylabel('Number of Eggs Per Ovitrap')
plt.legend()
plt.legend(loc='center', bbox_to_anchor=(.9, .1))
plt.grid(False)
plt.tight_layout()
plt.show()

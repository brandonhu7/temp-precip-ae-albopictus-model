# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 18:01:02 2025

@author: Brandon
"""

#this code fits h (hatching rate) to real data from Delatte et al. 
#it also plots all of the temperature-dependent parameters (h, lambda, nu, and mu_a)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv('hatching_rate.csv')


def temp_quadratic(T, T_0, T_M, c):
    return np.maximum(0, c*(T-T_0)*(T_M-T))

def temp_briere(T, T_0, T_M, c):
    return np.maximum(0, c*T*(T-T_0)*np.sqrt(np.maximum(T_M-T,0)))

temps = df['temps'].values
h = df['h'].values

initial_guess = [15,35,.003]
popt, pcov = curve_fit(temp_quadratic, temps, h, p0=initial_guess, bounds=([0, 0, 0], [45, 45, 1]))


def temp_lambda(T):
    return (np.maximum(-.3379*np.power(T,2) + 16.86*T - 142.8,0))/(2*(np.maximum(0.045*np.power(T,2) - 2.717*T + 44.41,0)))


x_fit = np.linspace(min(temps), max(temps), 300)
y_fit = temp_quadratic(x_fit, 12.99, 43.74, 0.000445) #h
y_fit1 = temp_quadratic(x_fit, 13.41, 31.51, 0.148) #mua
y_fit2 = temp_briere(x_fit, 8.60, 39.66, 0.0000638) #nu
y_fit3 = temp_lambda(x_fit) #lambda


plt.figure(figsize=(10,6))
plt.plot(x_fit, y_fit, label='Hatching Rate Curve - quadratic', color='limegreen')
plt.title('Hatching Rate vs Temperature', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Temperature (°C)', fontsize=20)
plt.ylabel('Hatching Rate (days⁻¹)', fontsize=20)
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(x_fit, y_fit1, label='Adult Mortality Rate Curve - quadratic', color='midnightblue')
plt.title('Adult Mortality Rate vs Temperature', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Temperature (°C)', fontsize=20)
plt.ylabel('Adult Mortality Rate (days⁻¹)', fontsize=20)
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(x_fit, y_fit2, label='Development Rate Curve - Briere', color='firebrick')
plt.title('Development Rate vs Temperature', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Temperature (°C)', fontsize=20)
plt.ylabel('Development Rate (days⁻¹)', fontsize=20)
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(x_fit, y_fit3, label='Fecundity Rate Curve - piecewise rational', color='gold')
plt.title('Fecundity Rate (eggs laid per adult female) vs Temperature', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Temperature (°C)', fontsize=20)
plt.ylabel('Fecundity Rate (days⁻¹)', fontsize=20)
plt.legend(fontsize=14)
plt.show()
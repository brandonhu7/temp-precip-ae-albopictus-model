# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 22:00:10 2025

@author: Brandon
"""


#fits temperature and precipitation data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#temp_precip_data used as example, can use subsets of weeks as well
df1 = pd.read_csv('wk1-141tp.csv')
df1 = df1[(df1['week'] >= 0) & (df1['week'] < 143)]

def model_function(x, a, c, d):
    return a*np.cos((2*np.pi/52)*x-c)+d

def asymmetric_cosine_function(x, A_top, A_bottom, c, d):

    y_raw = np.cos(0.122 * x + c)
    y = np.where(y_raw > 0, A_top * y_raw, A_bottom * y_raw)
    return y + d

def fourier(x, a0, a1, b1, a2, b2, a3, b3):
    w = 2*np.pi/52
    return (a0 + a1*np.cos(w*x) + b1*np.sin(w*x) + a2*np.cos(2*w*x) + b2*np.sin(2*w*x) + a3*np.cos(3*w*x) + b3*np.sin(3*w*x))


fourier_guess = [1,1,1,1,1,1,1]
initial_guess2 = [10,1,1.6]

initial_guess=[2.6, 1.6, 24]

average_temperature_per_week = df1.groupby('week')['tmean '].mean().reset_index()
max_temperature_per_week = df1.groupby('week')['tmax'].max().reset_index()
min_temperature_per_week = df1.groupby('week')['tmin'].min().reset_index()
precipitation_per_week = df1.groupby('week')['precipitation (mm)'].sum().reset_index()


weeks = average_temperature_per_week['week'].values
mean_temps = average_temperature_per_week['tmean '].values
max_temps = max_temperature_per_week['tmax'].values
min_temps = min_temperature_per_week['tmin'].values
precipitation = precipitation_per_week['precipitation (mm)'].values

popt, pcov = curve_fit(model_function, weeks, mean_temps, p0=initial_guess)
poptmax, pcovmax = curve_fit(model_function, weeks, max_temps, p0=initial_guess)
poptmin, pcovmin = curve_fit(model_function, weeks, min_temps, p0=initial_guess)
popt1, pcov1 = curve_fit(fourier, weeks, precipitation, p0=fourier_guess)

x_fit = np.linspace(min(weeks), max(weeks), 300)
y_fit = model_function(x_fit, *popt)
y_fitmax = model_function(x_fit, *poptmax)
y_fitmin = model_function(x_fit, *poptmin)
y_fit1 = fourier(x_fit, *popt1)

plt.figure(figsize=(10,6))
plt.scatter(weeks, mean_temps, label='Data', color='lightgreen')
#plt.plot(weeks, min_temps, label='Minimum Temperature by Week', color='blue')
#plt.plot(weeks, max_temps, label='Maximum Temperature by Week', color='red')
plt.plot(x_fit, y_fit, label='Fitted Curve', color='fuchsia',linewidth=2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.plot(x_fit, y_fitmax, label='Fitted Curve - Max Temps', color='green')
#plt.plot(x_fit, y_fitmin, label='Fitted Curve - Min Temps', color='green')
plt.title('Mean Temperature By Week', fontsize=20)
plt.xlabel('Time (weeks)', fontsize=20)
plt.ylabel('Temperature (°C)', fontsize=20)
plt.legend(fontsize=14, loc='upper left')
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(weeks, precipitation, label='Data', color='lightgreen')
plt.plot(x_fit, y_fit1, label='Fitted Curve', color='purple',linewidth=2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Total Precipitation By Week', fontsize=20)
plt.ylabel('Precipitation (mm)', fontsize=20)
plt.xlabel('Time (weeks)', fontsize=20)
plt.legend(fontsize=14, loc='upper left')
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:07:59 2026

@author: Brandon
"""

#plots temp and precip data variations

import numpy as np
import matplotlib.pyplot as plt

def model_function(x, a, c, d):
    return a*np.cos((2*np.pi/52)*x-c)+d

def asymmetric_cosine_function(x, A_top, A_bottom, c, d):

    y_raw = np.cos(0.122 * x + c)
    y = np.where(y_raw > 0, A_top * y_raw, A_bottom * y_raw)
    return y + d

def fourier(x, a0, a1, b1, a2, b2, a3, b3):
    w = 2*np.pi/52
    return (a0 + a1*np.cos(w*x) + b1*np.sin(w*x) + a2*np.cos(2*w*x) + b2*np.sin(2*w*x) + a3*np.cos(3*w*x) + b3*np.sin(3*w*x))

x_fit = np.linspace(0, 52, 300)
y_fit = model_function(x_fit, -2.78957, 1.48823, 24.5384)
y_fitmax = model_function(x_fit, -2.78957, 1.48823, 28.5384)
y_fitmin = model_function(x_fit, -2.78957, 1.48823, 20.5384)

y_fit2 = model_function(x_fit, -2.78957, 1.48823, 24.5384)
y_fit2max = model_function(x_fit, -0.78957, 1.48823, 24.5384)
y_fit2min = model_function(x_fit, -4.78957, 1.48823, 24.5384)

y_fit1 = fourier(x_fit, 27.5602, 1.49, -27.955, -13.243, -0.292826, 2.58997, 2.73886)
y_fit1max = fourier(x_fit, 37.5602, 1.49, -27.955, -13.243, -0.292826, 2.58997, 2.73886)
y_fit1min = fourier(x_fit, 7.5602, 1.49, -27.955, -13.243, -0.292826, 2.58997, 2.73886)


plt.figure(figsize=(10,6))
#plt.scatter(weeks, mean_temps, label='Data', color='lightgreen')
#plt.plot(weeks, min_temps, label='Minimum Temperature by Week', color='blue')
#plt.plot(weeks, max_temps, label='Maximum Temperature by Week', color='red')
plt.plot(x_fit, y_fit, label='Original', color='darkgreen',linewidth=2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(x_fit, y_fitmax, label='+4°C', color='firebrick',linewidth=2)
plt.plot(x_fit, y_fitmin, label='-4°C', color='slateblue',linewidth=2)
plt.xlabel('Time (weeks)', fontsize=20)
plt.ylabel('Temperature (°C)', fontsize=20)
plt.legend(fontsize=14, loc='upper left')
plt.show()

plt.figure(figsize=(10,6))
#plt.scatter(weeks, mean_temps, label='Data', color='lightgreen')
#plt.plot(weeks, min_temps, label='Minimum Temperature by Week', color='blue')
#plt.plot(weeks, max_temps, label='Maximum Temperature by Week', color='red')
plt.plot(x_fit, y_fit2, label='Original', color='darkgreen',linewidth=2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(x_fit, y_fit2max, label='+2°C', color='firebrick',linewidth=2)
plt.plot(x_fit, y_fit2min, label='-2°C', color='slateblue',linewidth=2)
plt.xlabel('Time (weeks)', fontsize=20)
plt.ylabel('Temperature (°C)', fontsize=20)
plt.legend(fontsize=14, loc='upper left')
plt.show()

plt.figure(figsize=(10,6))
#plt.scatter(weeks, precipitation, label='Data', color='lightgreen')
plt.plot(x_fit, y_fit1, label='Original', color='darkgreen',linewidth=2)
plt.plot(x_fit, y_fit1max, label='+10mm', color='firebrick',linewidth=2)
plt.plot(x_fit, y_fit1min, label='-20mm', color='slateblue',linewidth=2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Precipitation (mm)', fontsize=20)
plt.xlabel('Time (weeks)', fontsize=20)
plt.legend(fontsize=14, loc='upper left')
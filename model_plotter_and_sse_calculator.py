# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 21:17:28 2026

@author: Brandon
"""

#plots models in a well-formatted graph (also calculates sum of squared errors for any timespan)

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

#t_plotfull and tp_plotfull (full time period for both temp only and temp and precip models) used as an example, can use for any time period
df = pd.read_csv("t_plotfull.csv") 
df1 = pd.read_csv('tp_plotfull.csv')
merged = pd.read_csv('mosquitoes_by_week.csv')
#mosquitoes_by_week is the raw ovitrap data


df['squared_error'] = (df['calculated'] - df['real']) ** 2
df1['squared_error'] = (df1['calculated'] - df1['real']) ** 2


sse = df['squared_error'].sum()
sse1 = df1['squared_error'].sum()

tB = 1
tF = 141 
nTime = 141

#the timespan (adjust based on how long the timespan is)
tspan = np.linspace(tB,tF,nTime)

plt.figure(figsize=(10, 6))
plt.scatter(merged['week'], merged['individualCount'], label='Data ', color='indigo')
plt.plot(tspan, df['calculated'], label='Temperature Only Model Prediction', color='dodgerblue', linewidth=3)
plt.plot(tspan, df1['calculated'], label='Temperature and Precipitation Model Prediction', color='orangered', linewidth=3)
plt.xlabel('Time (weeks)', fontsize=14)
plt.ylabel('Number of Eggs Per Ovitrap', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.legend(loc='center', bbox_to_anchor=(.26, .91), fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

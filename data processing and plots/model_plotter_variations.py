# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:14:47 2026

@author: Brandon
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

#t_plotfull and tp_plotfull (full time period for both temp only and temp and precip models) used as an example, can use for any time period
df = pd.read_csv("variations_data.csv") 
#mosquitoes_by_week is the raw ovitrap data

tB = 1
tF = 52
nTime = 52

#the timespan (adjust based on how long the timespan is)
tspan = np.linspace(tB,tF,nTime)


plt.figure(figsize=(10, 6))

plt.plot(tspan, df['tempplus4'], label='+4°C', color='firebrick', linewidth=3)
plt.plot(tspan, df['tempplus3'], label='+3°C', color='firebrick', linewidth=3)
plt.plot(tspan, df['tempplus2'], label='+2°C', color='firebrick', linewidth=3)
plt.plot(tspan, df['tempplus1'], label='+1°C', color='firebrick', linewidth=3)
plt.plot(tspan, df['normalmt'], label='Original', color='darkgreen', linewidth=3)
plt.plot(tspan, df['tempminus1'], label='-1°C', color='slateblue', linewidth=3)
plt.plot(tspan, df['tempminus2'], label='-2°C', color='slateblue', linewidth=3)
plt.plot(tspan, df['tempminus3'], label='-3°C', color='slateblue', linewidth=3)
plt.plot(tspan, df['tempminus4'], label='-4°C', color='slateblue', linewidth=3)
plt.xlabel('Time (weeks)', fontsize=14)
plt.ylabel('Number of Eggs Per Ovitrap', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.legend(loc='upper left', fontsize=14, ncols=3)
plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.plot(tspan, df['ampplus2'], label='+2°C', color='firebrick', linewidth=3)
plt.plot(tspan, df['ampplus1'], label='+1°C', color='firebrick', linewidth=3)
plt.plot(tspan, df['normalta'], label='Original', color='darkgreen', linewidth=3)
plt.plot(tspan, df['ampminus1'], label='-1°C', color='slateblue', linewidth=3)
plt.plot(tspan, df['ampminus2'], label='-2°C', color='slateblue', linewidth=3)
plt.xlabel('Time (weeks)', fontsize=14)
plt.ylabel('Number of Eggs Per Ovitrap', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.legend(loc='upper left', fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(tspan, df['preplus10'], label='+10mm', color='firebrick', linewidth=3)
plt.plot(tspan, df['normalp'], label='Original', color='darkgreen', linewidth=3)
plt.plot(tspan, df['preminus10'], label='-10mm', color='slateblue', linewidth=3)
plt.plot(tspan, df['preminus20'], label='-20mm', color='slateblue', linewidth=3)
plt.xlabel('Time (weeks)', fontsize=14)
plt.ylabel('Number of Eggs Per Ovitrap', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.legend(loc='upper left', fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.show()

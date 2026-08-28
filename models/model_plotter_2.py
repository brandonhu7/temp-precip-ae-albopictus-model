# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:06:46 2026

@author: Brandon
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

#t_plotfull and tp_plotfull (full time period for both temp only and temp and precip models) used as an example, can use for any time period
df = pd.read_csv("test303.csv") 
#mosquitoes_by_week is the raw ovitrap data

tB = 1
tF = 52
nTime = 52

#the timespan (adjust based on how long the timespan is)
tspan = np.linspace(tB,tF,nTime)

plt.figure(figsize=(10, 6))
plt.plot(tspan, df['normal'], label='Original', color='darkgreen', linewidth=3)
plt.plot(tspan, df['preminus10'], label='-10mm', color='slateblue', linewidth=3)
plt.plot(tspan, df['preplus10'], label='+10mm', color='firebrick', linewidth=3)
plt.xlabel('Time (weeks)', fontsize=14)
plt.ylabel('Number of Eggs Per Ovitrap', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.legend(loc='center', bbox_to_anchor=(.12, .9), fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.show()
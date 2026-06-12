# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 20:27:44 2025

@author: Brandon
"""

#this assigns a week value (1-141) to every datapoint in order to make future work with this data easier
#also cleans up raw data from the dataset from Le Goff et al. 

from pygbif import registry
from pygbif import occurrences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_key = "3c4d3e45-4a7a-4ce6-8ead-6474004952c6"
dataset_info = registry.datasets(uuid=dataset_key)

all_results = []

for offset in range(0, 10000, 300):
    res = occurrences.search(datasetKey=dataset_key, hasCoordinate=True, hasGeospatialIssue=False,limit=300,offset=offset)
    all_results.extend(res['results'])

#dataframe with the entire imported dataset
df = pd.DataFrame(all_results)

#removing redundant columns to make code run faster
columns_to_keep = ['startDayOfYear', 'endDayOfYear', 'locationID','year','month','occurrenceID', 'individualCount']

filtered_df = df[columns_to_keep].copy()

del df


#organizing dataframe and assigning each entry a week value
filtered_df['week'] = 0

mask = filtered_df['year'] == 2013
filtered_df.loc[mask, 'week'] = np.floor(filtered_df.loc[mask, 'startDayOfYear'] / 7)-18


mask = filtered_df['year'] == 2013
filtered_df.loc[mask, 'week'] = np.floor(filtered_df.loc[mask, 'startDayOfYear'] / 7)-18


mask1 = filtered_df['year'] == 2014
filtered_df.loc[mask1, 'week'] = np.floor((filtered_df.loc[mask1, 'startDayOfYear']-3) / 7)+35


mask2 = filtered_df['year'] == 2015
filtered_df.loc[mask2, 'week'] = np.floor((filtered_df.loc[mask2, 'startDayOfYear']+1) / 7)+86


#manual additions for week
mask21 = filtered_df['startDayOfYear'] == 41 #these don't have "year" for some reason, but i found that they were in 2015 and calculated the week
filtered_df.loc[mask21, 'week'] = 90

mask22 = filtered_df['startDayOfYear'] == 97  #these don't have "year" for some reason, but i found that they were in 2015 and calculated the week
filtered_df.loc[mask22, 'week'] = 98

mask23 = (filtered_df['startDayOfYear'] == 363) | (filtered_df['startDayOfYear'] == 365) #transition between 2015 and 16 - also don't have "year"
filtered_df.loc[mask23, 'week'] = 136


mask3 = filtered_df['year'] == 2016
filtered_df.loc[mask3, 'week'] = np.floor((filtered_df.loc[mask3, 'startDayOfYear']-1) / 7)+139

filtered_df = filtered_df[filtered_df['week'] != 0]

average_mosquitoes_per_week = filtered_df.groupby('week')['individualCount'].mean().reset_index()



all_weeks_df = pd.DataFrame({'week': range(1, 142)})


merged = pd.merge(all_weeks_df, average_mosquitoes_per_week, on='week', how='left')
merged['individualCount'] = merged['individualCount'].interpolate(method='linear')

merged.to_csv("wk1-141m.csv", index=False)

plt.figure(figsize=(10,6))
plt.plot(merged['week'], merged['individualCount'], label='Average mosquitoes per week', color='blue')
plt.title('Average Number of Mosquitoes Per Ovitrap By Week', fontsize=14)
plt.xlabel('Weeks (since May 11, 2013)', fontsize=12)
plt.ylabel('Average Number of Mosquitoes Per Ovitrap', fontsize=12)
plt.legend()
plt.show()
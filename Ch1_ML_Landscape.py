# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 22:57:17 2018

@author: vamsi.mudimela
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import os

os.chdir('C:\\Users\\vamsi.mudimela\\Documents\\Library\\learning\\Geron\\data')

df_life_sat = pd.read_csv('BLI_04072018074322010.csv')
df_life_sat = df_life_sat[df_life_sat['Indicator'] == 'Life satisfaction'] 

df_gdp_per_cap = pd.read_csv('gdp_per_cap.csv', encoding='latin1')


df_merged = pd.merge(df_life_sat[['Country','Value']], 
                     df_gdp_per_cap[['Country','2015']], 
                     how='inner', on='Country')

df_merged.rename(columns={'Value':'Life Satisfaction', '2015':'GDP per capita'}, inplace=True)
df_merged['GDP per capita'] = df_merged['GDP per capita'].str.replace(',','')
df_merged['GDP per capita'] = df_merged['GDP per capita'].astype(float) 

x = np.c_[df_merged['GDP per capita']]
y = np.c_[df_merged['Life Satisfaction']]

df_merged.plot(kind='scatter',x='GDP per capita', y='Life Satisfaction')
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(x,y)

x_new = [[22587]]
print(lin_reg.predict(x_new))

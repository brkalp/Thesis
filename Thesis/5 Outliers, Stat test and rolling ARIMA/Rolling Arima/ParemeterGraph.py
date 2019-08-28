# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:33:24 2019

@author: Qognica
"""
from pmdarima.arima import auto_arima
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime 
import seaborn as sns
sns.set()
data0 = pd.DataFrame(web.DataReader("SWED-A.ST", "yahoo", start = "2000", end = datetime.datetime(2019, 4, 11))['Adj Close'])
# Beginning = len(data0)*0.7
# Each Split =  int(len(data0)*0.7/300) # Each split is around 12 data entries.
plt.plot(data0.values)
data0 = data0[int(-len(data0)*0.3):]
df2 = pd.DataFrame(columns = ['Price'],index = range(0,300))
for i in range(0,300):
    split = 0.7 + i*0.001
    #print(int(split*len(data0)))
    df2['Price'][i] = data0.iloc[ int( split * len(data0) ) ]['Adj Close']

    
import os
os.chdir(r'C:\Users\Qognica\Dropbox\Qognica Master Thesis 2019\Work folder Alp\alp\Written Thesis reports\8 Rolling ARIMA identification\300 for orders and parameters')
Phist = pd.read_csv(r'orders.txt').iloc[:,1:] # is written parameter index.
# plt.plot(Phist['1'][-300:]) AR
# plt.plot(Phist['2'][-300:]) MA
AR = pd.read_csv(r'coefar.txt').iloc[:,1:]
MA = pd.read_csv(r'coefma.txt').iloc[:,1:]
plt.plot(AR[-300:], label = 'AR multipliers',color = 'k')
plt.plot(MA[-300:], label = 'MA multipliers');plt.title('AR and MA multipliers of model over time');plt.legend();plt.show()

df = pd.DataFrame(columns = ['AR','MA','ARMA','ARMA diff'], index = AR.index)
df['AR'] = AR; df['MA'] = MA; df['ARMA'] = df['AR'] + df['MA']; df['ARMA diff'] = df['ARMA'] - df['ARMA'].shift(1)
df = df[-300:]; df['Counter'] = pd.Series([i for i in range(300,600)], index = df.index)
df.index = data0.index
plt.plot(df['ARMA diff'][1:])
df['Price'] = data0

fig, ax1=plt.subplots()
plt.plot(df.index,df2,'k',label = 'Swedbank Last 50 days')
ax2 = ax1.twinx()
plt.plot(df['ARMA diff'][1:],label = 'Parameters change', alpha = 0.5)
plt.legend();plt.title('Stock Price vs Differences in parameters')


"""
df3= pd.DataFrame(index = df.index)
df3['Param'] = df['ARMA diff']
df3['Price'] = df2.pct_change().values
df3 = df3.dropna()
sns.pairplot(df3, kind="reg")

data0 = data0[-len(Phist):]
#for i in range(6,12):plt.plot(Phist[(f'{list(Phist)[i]}')])
df = pd.DataFrame(index = Phist.index)
df['AR'] = Phist['AR1'] + Phist['AR2'] + Phist['AR3'] + Phist['AR4'] + Phist['AR5']
df['MA'] = Phist['MA1'] + Phist['MA2'] + Phist['MA3'] + Phist['MA4'] + Phist['MA5']
plt.plot(df['AR'],'k',label = 'Sum of all AR Parameters');plt.title('Sum of AR coefficients');plt.legend();plt.show()
plt.plot(df['MA'],'k',label = 'Sum of all MA Parameters');plt.title('Sum of MA coefficients');plt.legend();plt.show()
df['both'] = df['AR']  +  df['MA']

plt.plot(df['both'],'k',label = 'Sum of all Parameters');plt.title('Sum of both coefficients');plt.legend()
df['bothdiff'] = df['both'] - df['both'].shift(1)

df.index = data0.index
fig, ax1=plt.subplots()
plt.plot(data0,'k',label = 'Swedbank Last 50 days')
ax2 = ax1.twinx()
plt.plot(df['AR'],label = 'Difference of the parameters sum', alpha = 0.5);plt.title('Stock Price vs Parameter Differences')

plt.plot(Phist['ARorders'])






fig, ax1=plt.subplots()
outliers = abs(data0.pct_change() / np.std(abs(data0.pct_change()))).apply(np.floor).dropna()

from datetime import timedelta
for i in outliers[outliers>2].dropna().index: ax1.axvspan(i-timedelta(days=+50), i+timedelta(days=+50), alpha=0.04, color='red')
plt.plot(ARsum['ARdiff'],'k',label = 'Percentage Returns')
"""
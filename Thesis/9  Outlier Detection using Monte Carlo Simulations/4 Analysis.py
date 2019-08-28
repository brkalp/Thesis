import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas_datareader as web
import seaborn as sns
sns.set()
from datetime import timedelta
import time
import os
#os.getcwd()
os.chdir(r"C:\Users\ripintheblue\Desktop\OutlierWmc")
TT = time.time()
Ticker = "SWED-A.ST"
data = web.DataReader(Ticker, 'yahoo', "2010" ,'2019-06-23' )['Adj Close'];

Scoring = pd.read_csv(r'Scoring.csv').set_index('Date').dropna()
Scoring.index = data.iloc[1:-1].index
plt.plot(Scoring['Adj Close']);plt.show()
plt.plot(Scoring['Returns']);plt.show()
plt.plot(Scoring['ActPreds']);plt.show()
plt.plot(Scoring['AvgPreds']);plt.show()
Scoring['Diff'] = - Scoring['ActPreds'] + Scoring['AvgPreds']

plt.plot(Scoring['Diff']);plt.show()
plt.hist(Scoring['Diff'],bins = 50)
#Scoring['Diff'].mean()
#Scoring['Diff'].median()
#Scoring['Diff'].std()
Scoring['Diff'].describe()
 
print('Max error:', Scoring[Scoring['Diff'] == Scoring['Diff'].max()])
print('\nMin error:', Scoring[Scoring['Diff'] == Scoring['Diff'].min()])

print(len(Scoring[Scoring['Diff'] < -0.1]))
print(len(Scoring[Scoring['Diff'] > 0.1]))

print(Scoring[Scoring['Diff'] < -0.1])
print(Scoring[Scoring['Diff'] > 0.1])


plt.hist(Scoring['Diff'],bins = 50);plt.title('ANN output errors');plt.show()
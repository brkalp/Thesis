import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Getting data
Ticker = "SWED-A.ST"
data = web.DataReader(Ticker, 'yahoo', "2010")['Adj Close']; 
data = pd.DataFrame(data,index=data.index)
data['Returns'] = data.pct_change()
data = data.dropna()
data.name = Ticker
# data['Returns'].mean()
sigmamultiplier = 5
Limit = data['Returns'].std() * sigmamultiplier
data['Dev'] =  abs(data['Returns'] - data['Returns'].mean()) / data['Returns'].std()
data['DevFloor'] = data['Dev'].apply(np.floor)

# Plot returns with the limit dev
plt.plot(abs(data['Returns']),linestyle='--',color='k',linewidth=0.5);plt.grid();plt.title('%s Percent Returns, Filter:%i Sigma'%(data.name, sigmamultiplier))
plt.axhline(y = Limit ,color='r',linestyle='--',linewidth=1);plt.grid();plt.title('%s Percent Returns, Filter:%i Sigma'%(data.name, sigmamultiplier))
plt.show()

# Plot Stock Price with Outliers shaded red
Outliers = pd.DataFrame(data['Dev'][data['Dev']>sigmamultiplier],data['Dev'][data['Dev']>sigmamultiplier].index)
fig, ax1=plt.subplots()
plt.plot(data['Adj Close'],linewidth = 0.5,color='k',label = data.name)
plt.title('Detected %i Outliers above %i Sigma Limit'%(len(Outliers),sigmamultiplier))
ax2 = ax1.twinx();plt.grid()
timerange = timedelta(days=+1)
for date in Outliers.index: ax1.axvspan(date-timerange, date+timerange, alpha=0.3, color='red')
plt.show()

# Plot data with Outlier Levels
fig, ax1=plt.subplots()
plt.plot(data['Adj Close'],color='k',label = data.name,linewidth=0.5)
ax2 = ax1.twinx()
plt.plot(data['Dev'][data['Dev']>sigmamultiplier], 's', markersize= 4,color='r',label = 'Outlier Levels, inputs to neural network')
plt.yticks(range(0,10));plt.grid();plt.title('Outlier Levels');plt.legend()
plt.show()
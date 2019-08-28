import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
This function tests the null hypothesis that a sample comes from a normal distribution. 
Based on D’Agostino and Pearson’s. Test based on skewness and kurtosis and will miss bimodality

Augmented Dickey-Fuller
The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, 
with the alternative that there is no unit root. 
If the pvalue is above a critical size, then we cannot reject that there is a unit root.
p val = 0 stationary.
"""
#S&P500 data
data0 = pd.DataFrame(web.DataReader("SWED-A.ST", "yahoo", start = "2010", end = "2019")['Adj Close'])
data = data0.copy()
dataret = data.pct_change().dropna()

plt.plot(data);
#plt.plot(dataret)
def getnewseries(data,datarets):
    newdata = pd.DataFrame(columns = ['Adj Close'],index = data.index)
    newdata.iloc[0] = data.iloc[0].values
    for i in range(1,len(data)): newdata.iloc[i] = newdata.iloc[i-1] + newdata.iloc[i-1]*datarets.iloc[i-1]
    return newdata
plt.plot(getnewseries(data,dataret));plt.show()

window = 250
bot = -0.17
top = 0.12
"""
normresults = pd.Series()
for i in range(startingrange,(endingrange-window)):
    index = data.index[i]
    normresults = normresults.append(pd.Series(stats.normaltest(data[i:i+window])[1],index = [index]))
fig, ax1=plt.subplots(figsize=figsize)
plt.plot(data,'k',label = 'Data'); plt.legend(loc = 1);ax2 = ax1.twinx()
plt.plot(normresults,label = 'Norm test results', linewidth=0.5);plt.legend(loc = 1)
plt.grid();plt.title('Normality Test Results with Window: %i'%window) ;plt.show()
"""
############## First Graph to Show outliers effect on the stationarity tests
data = data0.copy()
dataret = data.pct_change().dropna()
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats
# Calculating Adfullers
adfuresults = pd.Series(index = data.index)
for i in range(1,(len(data)-window)):
    adfuresults[i+window] = adfuller(data.iloc[i:i+window,0].values)[1]
# Ready to Plot
fig, ax1=plt.subplots();plt.plot(dataret,'k',label = 'Percentage Returns');plt.ylim(bot,top)
ax2 = ax1.twinx();plt.plot(adfuresults,label = 'Adfuller test results on price',alpha=0.8,linewidth = 1); plt.legend(loc = 1)
# Putting The red marks
outliers = abs(dataret / np.std(abs(dataret))).apply(np.floor);outliers = outliers[outliers>2].dropna()
from datetime import timedelta
for i in outliers.index: ax1.axvspan(i-timedelta(days=+50), i+timedelta(days=+50), alpha=0.04, color='red')
plt.title('Adfuller test results with a %i rolling window and Outliers shaded in Red'%window);plt.grid()
plt.show()

########### without  2 outliers
data3 = data0.copy()
dataret3 = data3.pct_change().dropna()
outliers = abs(dataret / np.std((dataret3))).apply(np.floor);outliers = outliers[outliers>3].dropna()
dataret3.loc[outliers.index] = 0
plt.plot(data3, 'k',label='Swedbank');plt.plot(getnewseries(data3,dataret3), label='Swedbank without outliers')
plt.title('Swedbank Price excluding big outliers in Percentage Return');plt.legend(loc = 1);plt.show()
data2 = getnewseries(data3,dataret3)
data2rets = data2.pct_change().dropna()
# Calculating Adfullers
adfuresults2 = pd.Series(index = data2.index)
for i in range(1,(len(data3)-window)):
    adfuresults2[i+window] = adfuller(data2.iloc[i:i+window,0].values)[1]
# Calculating Adfullers for returns
adfuresults2rets = pd.Series(index = data2.index)
for i in range(1,(len(data3)-window)):
    adfuresults2rets[i+window] = adfuller(data2rets.iloc[i:i+window,0].values)[1]
# Ready to Plot
fig, ax1=plt.subplots();plt.plot(data2rets,'k',label = 'Percentage Returns');plt.ylim(bot,top)
ax2 = ax1.twinx();plt.plot(adfuresults2,label = 'Adfuller test results on new price',alpha=0.8,linewidth = 1); plt.legend(loc = 1)
plt.plot(adfuresults2rets,label = 'Adfuller test results on new returns',alpha=0.8,linewidth = 1); plt.legend(loc = 1)
# Putting The red marks
outliers = abs(data2rets / np.std((dataret))).apply(np.floor);outliers = outliers[outliers>2].dropna()
from datetime import timedelta
for i in outliers.index: ax1.axvspan(i-timedelta(days=+50), i+timedelta(days=+50), alpha=0.04, color='red')
plt.title('3 sigma excluded, Adfuller test results with a %i rolling window'%window);plt.grid()
plt.show()


########### without  2 outliers
data4 = data0.copy()
dataret4 = data4.pct_change().dropna()
outliers = abs(dataret / np.std((dataret4))).apply(np.floor);outliers = outliers[outliers>2].dropna()
dataret4.loc[outliers.index] = 0
plt.plot(data4, 'k',label='Swedbank');plt.plot(getnewseries(data4,dataret4), label='Swedbank without outliers')
plt.title('Swedbank Price excluding big outliers in Percentage Return');plt.legend(loc = 1);plt.show()
data5 = getnewseries(data4,dataret4)
data5rets = data5.pct_change().dropna()
# Calculating Adfullers
adfuresults2 = pd.Series(index = data5.index)
for i in range(1,(len(data4)-window)):
    adfuresults2[i+window] = adfuller(data5.iloc[i:i+window,0].values)[1]
# Calculating Adfullers for returns
adfuresults2rets = pd.Series(index = data5.index)
for i in range(1,(len(data4)-window)):
    adfuresults2rets[i+window] = adfuller(data5rets.iloc[i:i+window,0].values)[1]
# Ready to Plot
fig, ax1=plt.subplots();plt.plot(data5rets,'k',label = 'Percentage Returns');plt.ylim(bot,top)
ax2 = ax1.twinx();plt.plot(adfuresults2,label = 'Adfuller test results on new price',alpha=0.8,linewidth = 1); plt.legend(loc = 1)
plt.plot(adfuresults2rets,label = 'Adfuller test results on new returns',alpha=0.8,linewidth = 1); plt.legend(loc = 1)
# Putting The red marks
outliers = abs(data5rets / np.std((dataret))).apply(np.floor);outliers = outliers[outliers>2].dropna()
from datetime import timedelta
for i in outliers.index: ax1.axvspan(i-timedelta(days=+50), i+timedelta(days=+50), alpha=0.04, color='red')
plt.title('2 sigma excluded, Adfuller test results with a %i rolling window'%window);plt.grid()
plt.show()



plt.plot(data0, 'k',label='Swedbank Price')
plt.plot(data2,label='Without 3 Sigma Outliers')
plt.plot(data5,label='Without 2 Sigma Outliers');plt.title('Swedbank Price without outliers in returns'); plt.legend(loc = 1);plt.show()
print('P val from sample %3.3f'%adfuller(data0.iloc[i:i+window,0].values)[1])
print('P val without 3 sigma outliers %3.3f'%adfuller(data2.iloc[i:i+window,0].values)[1])
print('P val without 2 sigma outliers %3.3f'%adfuller(data5.iloc[i:i+window,0].values)[1])

print('P val from sample %3.8f'%adfuller(data0.pct_change().dropna().iloc[i:i+window,0].values)[1])
print('P val without high outliers %3.8f'%adfuller(data2.pct_change().dropna().iloc[i:i+window,0].values)[1])
print('P val without all outliers %3.8f'%adfuller(data5.pct_change().dropna().iloc[i:i+window,0].values)[1])





"""
############## Second graph without Outliers
data1 = abs(dataret / np.std(abs(dataret))).apply(np.floor);data1 = data1[data1>0].dropna()
data2 = data.drop(data1.index)
adfuresults2 = pd.Series(index = data.index)
for i in range(1,(len(data)-window)):
    adfuresults2[i+window] = adfuller(data2.iloc[i:i+window,0].values)[1]
# Ready to Plot
fig, ax1=plt.subplots();plt.plot(data2,'k',label = 'Data');plt.ylim(bot,top)
ax2 = ax1.twinx();plt.plot(adfuresults2,label = 'Adfuller test results on price excluding outliers'); plt.legend(loc = 1)
# Putting The red marks
data1 = abs(data2 / np.std(abs(data))).apply(np.floor);data1 = data1[data1>0].dropna()
from datetime import timedelta
for i in data1.index: ax1.axvspan(i-timedelta(days=+50), i+timedelta(days=+50), alpha=0.03, color='red')
plt.title('Adfuller test results with a %i rolling window and Outliers shaded in Red'%window);plt.grid()

##############
############## Normality
##############

############## First Graph to Show outliers effect on the Normality tests
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats
normresults = pd.Series(index = data.index)
for i in range(1,(len(data)-window)):
    normresults[i+window] = stats.normaltest(data.iloc[i:i+window,0].values)[1]

fig, ax1=plt.subplots();plt.plot(data,'k',label = 'Data');bot,top = plt.ylim()
ax2 = ax1.twinx();plt.plot(normresults,label = 'Norm test results'); plt.legend(loc = 1)
data1 = abs(data / np.std(abs(data))).apply(np.floor);data1 = data1[data1>1].dropna()
from datetime import timedelta
for i in data1.index: ax1.axvspan(i-timedelta(days=+50), i+timedelta(days=+50), alpha=0.02, color='red')
plt.title('Adfuller test results with a %i rolling window and Outliers shaded in Red'%window);plt.grid()
plt.show()

############## Second graph without Outliers
data1 = abs(data / np.std(abs(data))).apply(np.floor);data1 = data1[data1>0].dropna()
data2 = data.drop(data1.index)
normresults = pd.Series(index = data2.index)
for i in range(1,(len(data2)-window)):
    normresults[i+window] = stats.normaltest(data2.iloc[i:i+window,0].values)[1]

fig, ax1=plt.subplots();plt.plot(data2,'k',label = 'Data');plt.ylim(bot,top)
ax2 = ax1.twinx();plt.plot(normresults,label = 'Adfuller test results excluding outliers'); plt.legend(loc = 1)
data1 = abs(data2 / np.std(abs(data))).apply(np.floor);data1 = data1[data1>0].dropna()
from datetime import timedelta
for i in data1.index: ax1.axvspan(i-timedelta(days=+50), i+timedelta(days=+50), alpha=0.03, color='red')
plt.title('Adfuller test results with a %i rolling window and Outliers shaded in Red'%window);plt.grid()
"""
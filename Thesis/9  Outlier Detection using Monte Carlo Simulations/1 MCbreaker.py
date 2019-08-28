import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.install_packages('quantmod')
qmod = rpackages.importr('quantmod')
utils.install_packages('forecast')
fcast = rpackages.importr('forecast')
import rpy2.robjects as robjects
autoarima = robjects.r['auto.arima']
simulate = robjects.r['simulate']
arima = robjects.r['arima']
coef = robjects.r['coef']
summary = robjects.r['summary']
asnumeric = robjects.r['as.numeric']
from rpy2.robjects import pandas2ri
Arima = robjects.r['Arima']

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas_datareader as web
import seaborn as sns
sns.set()
from datetime import timedelta

import time
TT = time.time()

Ticker = "SWED-A.ST"
data = web.DataReader(Ticker, 'yahoo', "2010" ,'2019-06-23' )['Adj Close'];
SIMSTART = -90; SIMLEN = 50; BREAKTIME = 15; UPPERSIGMA = 12; SIMTIM = 100

INPUT = data[:SIMSTART].copy()
#plt.plot(data.pct_change())
#plt.hist(data.pct_change(),bins = 50);
def getArimaModel(INPUT, SIMSTART):
    modelR0 = autoarima(pandas2ri.py2ri(INPUT))
    return modelR0
modelR0 = getArimaModel(INPUT,SIMSTART)
#print(coef(modelR0))
#print(summary(modelR0))

montecarlo0 = np.transpose([np.array(asnumeric(simulate(modelR0,nsim = SIMLEN))) for i in range(SIMTIM)])
montecarlo0 = pd.DataFrame( montecarlo0, index=[INPUT.index[-2] + timedelta(days = + i) for i in range(0, SIMLEN)] )
#plt.plot(INPUT[-150 - SIMSTART:],  linewidth = '1', label ='Data used for Calibration')
fig, ax = plt.subplots()
plt.plot(data[-150:], color = 'k', linewidth = '1', label ='Actual Swedbank Price')
plt.plot(montecarlo0.iloc[:,0:20],linestyle ='--', color = 'r', linewidth = '0.2')
plt.plot(montecarlo0.iloc[:,0:1],linestyle ='--', color = 'r', linewidth = '0.5',label = 'Simulations')
plt.plot(montecarlo0.mean(axis = 1), color = 'r', linestyle ='--', linewidth = '2',label = 'Mean of paths')
plt.title('MC Simulations');plt.legend()
fig.autofmt_xdate();plt.show()

def breakthis(orgpath, BREAKTIME, BREAKAMOUNT, model):
    orgpath = orgpath
    path = orgpath.iloc[0:BREAKTIME].copy()                                       # Cut from the breaking point
    path[-1] = path[-1] + BREAKAMOUNT * path[-1]
    INDEX = orgpath.iloc[BREAKTIME:].index                                # Get the index after breaking point
    adj_m = Arima(pandas2ri.py2ri(path),model=model)                        # Get model properties
    newpath = np.array(asnumeric(simulate(adj_m,nsim = len(INDEX))))        # Generate new path
    path = path.append( pd.DataFrame(newpath ,index = INDEX)) # Apend for return
    return path
    
SIGMATIMES = 0
Cumulative = montecarlo0.copy()
T0 = time.time()
for counter in range(0,UPPERSIGMA):
    SIGMATIMES = SIGMATIMES + 1
    print("Breaking, Sigma = %s, time = %s secs" %(str(SIGMATIMES).ljust(3),str(int(time.time() - T0)).ljust(3)) )
    # Simulating from new place
    montecarlo2 = montecarlo0.copy()
    BREAKAMOUNT = - data.pct_change().mean() - SIGMATIMES * data.pct_change().std()
    for i in range(SIMTIM): 
        BREAKTIME = 5 + i % (SIMLEN -5)
        montecarlo2[i] = breakthis(montecarlo0[i], BREAKTIME, BREAKAMOUNT, modelR0)
    Cumulative = pd.concat([Cumulative,montecarlo2], axis = 1)
   
#plt.plot(INPUT[-150 - SIMSTART:],  linewidth = '1', label ='Data used for Calibration')
fig, ax = plt.subplots()
plt.plot(data[-150:], color = 'k', linewidth = '1', label ='Actual Swedbank Price')
plt.plot(montecarlo2.iloc[:,:30],linestyle ='--', linewidth = '0.2', color = 'r')
plt.plot(montecarlo2.mean(axis = 1), color = 'r', linestyle ='--', linewidth = '2',label = 'Mean of MC')
plt.plot(montecarlo2.iloc[:,0:1],linestyle ='--', color = 'r', linewidth = '0.5',label = 'Simulations')
plt.legend; plt.title('Adjusted ARIMA Simulations with drop amount = %i sigma' %SIGMATIMES);plt.legend()
fig.autofmt_xdate();plt.show()
    

Cumulative.columns = [int(i/SIMTIM) for i in range((SIGMATIMES+1) * SIMTIM)]

plt.plot(data[-150:-70], linewidth = '1', color = 'k', label ='Actual Swedbank Price')
for i in range(12,-1,-5): plt.plot(Cumulative[i].mean(axis = 1), linestyle ='--', linewidth = '2',label = 'Mean of MC w sigma: %i'%(i) )
plt.legend(); plt.title('Simulation of breaking points');plt.legend();plt.show()

Cumulativerets = Cumulative.pct_change().dropna()
#Y = pd.DataFrame(0 , columns = Cumulativerets.columns, index = Cumulativerets.index)
#print ("BreakPoint on : ",Y[BREAKTIME-2:BREAKTIME -1].index)
#for i in Y.columns.drop_duplicates(): Y.loc[ Y[BREAKTIME-2:BREAKTIME -1].index, i] = i
Cumulativerets = Cumulative.pct_change().dropna()

#txt = input("Save csv (YES?): ")
#if txt =='YES':
#    print('Saved')
#    Cumulativerets.to_csv('Cumulativerets.csv')
#    

import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import pandas_datareader as web
import seaborn as sns
sns.set()
dataset = pd.DataFrame(web.DataReader("SWED-A.ST", "yahoo", start = "2010")['Adj Close'])
plt.plot(dataset['Adj Close'],label = 'SWED-A.ST', color = 'k', linewidth = 1); plt.legend(); plt.show()

########## AR1 example
np.random.seed(100)
sigma = 1; size = 100; alpha1 = 1.1
error=np.random.randn(size)*sigma ; x=np.zeros(size)
for i in np.arange(1,size): x[i] = alpha1*x[i-1] + error[i]
plt.plot(x[1:10], label='AR(1) process with $\u03C6_1$ = %2.2f' %(alpha1));plt.legend()

########## AR2 example
x=np.zeros(size); alpha2 = 0.7
for i in np.arange(2,size): x[i] = alpha1*x[i-1] + alpha2*x[i-2] + error[i]
plt.plot(x[1:10], label='AR(2) process with $\u03C6_1$ = %2.2f and $\u03C6_2$ = %2.2f' %(alpha1,alpha2));plt.legend();plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(x,lags = 10);plt.show()

########## ARMA example
from statsmodels.tsa.arima_process import arma_generate_sample
import statsmodels as sm
np.random.seed(100)
alpha1 = -0.5
#alpha2 = -0.5
ma1 = 1.1
ma2 = 0.5
ar = [1, alpha1]
ma = [1, ma1, ma2]
y = arma_generate_sample(ar, ma, 10000)
plt.plot(arma_generate_sample(ar, ma, 100), color = 'k', linewidth = '1',
         label='ARMA process $\u03C6_1$ = %2.1f, $\u03B8_1$ = %2.1f, $\u03B8_2$ = %2.1f' %(-alpha1, ma1, ma2));plt.legend();plt.show()

########## ARIMA Model Fit
from pmdarima.arima import auto_arima
model = auto_arima(y,  seasonality=False, trace=True,
                           error_action='ignore',  # don't want to know if an order does not work
                           suppress_warnings=True,  # don't want convergence warnings
                           stepwise=True)
model.summary()
model.params()

########## ARIMA example Residual Check ACF, Normplot and Stat Test
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(model.resid()[10:],lags = 50);plt.show()
def normalitytest(data0,alpha=0.05):
    """
    null hypothesis: x comes from a normal distribution
    statistic : float or array
    s^2 + k^2, where s is the z-score returned by skewtest and k is the z-score returned by kurtosistest.
    pvalue : float or array
    A 2-sided chi squared probability for the hypothesis test.
    
    x = stats.norm.rvs(size = 1000)
    normalitytest(x)
    """
    if isinstance(data0, pd.DataFrame):
        data = data0.iloc[:,0].values
    else:
        data = data0
    Normtest = stats.normaltest(data)
    print('  NT:[Stat&Pval:%6.2f,%6.2f' %(Normtest[0],Normtest[1]), end="")
    if Normtest[1]<alpha:
        print(':Norm= FF ]', end="")    
    if Normtest[1]>alpha:
        print(':Norm=TRUE]', end="")
normalitytest(model.resid()[10:])
from scipy import stats
def normplot(data):
    plt.hist(data, bins = 40, normed=True)
    mean, std = stats.norm.fit(data)                        # Get mean and standard deviation for Distribution input
    lin = np.linspace(min(plt.xticks()[0]), max(plt.xticks()[0]) , len(data)) # Min and max ticks for computing dist to be plotted
    Gen_dist = stats.norm.pdf(lin, mean, std)               # Generate Distribution
    plt.plot(lin, Gen_dist);plt.show()
    # normalitytest(model.resid()[10:]) 
normplot(model.resid()[10:])

from statsmodels.tsa.stattools import adfuller
def DickeyFuller(data, returnme = 'all'):
    header = ['Test Statistic','p-value','#Lags Used','Number of Observations Used']
    results = adfuller(data, autolag='AIC')
    dfoutput = pd.Series(results[0:4],index=header)
    if returnme == 'all':
        for key,value in results[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        return dfoutput
def DickeyFullerPrint(data0):
    if isinstance(data0, pd.DataFrame):
        data = data0.iloc[:,0].values
    else:
        data = data0
    Dickey = DickeyFuller(data, returnme = 'all')
    print('DF:[TestStat&CritVal:%6.2f,%6.2f,P-val:%4.2f' %(Dickey[0],Dickey[4],Dickey[1]), end="")
    if Dickey[0]>Dickey[4]:
        print(':Stat= FF ]', end="")    
    if Dickey[0]<Dickey[4]:
        print(':Stat=TRUE]', end="")
DickeyFullerPrint(model.resid()[10:])

######
###### Swedbank Model Fit
######

model = auto_arima(dataset,  seasonality=False, trace=True,
                           error_action='ignore',  # don't want to know if an order does not work
                           suppress_warnings=True,  # don't want convergence warnings
                           stepwise=True)
model.summary()
model.params()
#predictions= model.predict(n_periods=20)
#plt.plot(predictions)
normplot(model.resid()[10:],)
normalitytest(model.resid()[10:])
plot_acf(model.resid()[10:],lags = 50);plt.show()
DickeyFullerPrint(model.resid()[10:])
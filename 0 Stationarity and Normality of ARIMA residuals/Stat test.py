import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.tsa.stattools import adfuller
# getting data
data0 = web.DataReader("ABB", 'yahoo', "2016","2017")['Adj Close']

data = data0[:'2016-03-30']
OutofSampleData = data0['2016-03-31':'2016-07-30']

import seaborn as sns
sns.set()


#fig, ax1=plt.subplots(figsize=figsizes)
#plt.plot(data,color='k',linewidth = 1,label='ABB');plt.title('ABB Stock ARIMA fit');plt.show()
def DickeyFuller(data, returnme = 'all'):
    #https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
    """
    Null Hypothesis: The series has a unit root (value of a =1)
    Alternate Hypothesis: The series has no unit root.
    
    If the test statistic is less than the critical value, 
    we can reject the null hypothesis (aka the series is stationary). 
    When the test statistic is greater than the critical value, 
    we fail to reject the null hypothesis (which means the series is not stationary).
    """
    #Critical values 1,5,10 correspond for for 4,5,6
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

import scipy.stats as stats
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
        
def plotwithslices(data,figsize=(14,4),slices=5, RollingDistance=20,rollingwidth=1,stdwidth = 1, Testoutput = 0, doubleaxis=True):
    """
    Slice the data and prefirm DickeyFuller and scpi norm test
    """
    print("\n -- Plotting Data with %i Slices and RollingDist of %i and %i-- "  %(slices,RollingDistance,RollingDistance*5))
    totallen=len(data)
    slicelens=int(totallen/slices)
    fig, ax1=plt.subplots(figsize=figsize)
    plt.plot(data,label = 'Data',linewidth = 2)
    if (doubleaxis==True):
        ax2 = ax1.twinx()
    for i in range(slices-1):    
        plt.axvline(x=data.index[int(totallen/slices)*(i+1)],color='r',linestyle='--',linewidth=1)
    plt.legend();
    plt.show()
    
    if (Testoutput == True):
        print(' Slice  0 S: %4i E: %4i  ' %(0, totallen), end="")
        DickeyFullerPrint(data[0:totallen])
        normalitytest(data[0:totallen])
        for i in range(slices):   
            print('\n Slice %2i S: %4i E: %4i  ' %(i+1, slicelens*(i), slicelens*(i+1)), end="")
            DickeyFullerPrint(data[slicelens*(i):slicelens*(i+1)])
            normalitytest(data[slicelens*(i):slicelens*(i+1)])
    return

plotwithslices(data,slices=5,Testoutput = 1)
plotwithslices(data.pct_change().dropna(),slices=5,Testoutput = 1)

from pmdarima.arima import auto_arima
model = auto_arima(data, seasonality= False,   #trace=True,
                       error_action='ignore',  # don't want to know if an order does not work
                       suppress_warnings=True,  # don't want convergence warnings
                       stepwise=True,trace=True)
model.summary()



def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Model checking by residuals')
        smt.graphics.plot_acf(y, lags=10, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=10, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 

tsplot(model.resid()[10:])





from scipy import stats  
# create some normal random noisy data
ser = model.resid()[10:]
figsizes = (5,3)
fig, ax1=plt.subplots(figsize=figsizes)
# plot normed histogram
plt.hist(ser, bins = 25, normed=True)

# find minimum and maximum of xticks, so we know
# where we should compute theoretical distribution
xt = plt.xticks()[0]  
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(ser))

# lets try the normal distribution first
m, s = stats.norm.fit(ser) # get mean and standard deviation  
pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
plt.plot(lnspc, pdf_g, label="Norm") # plot it

plt.show()

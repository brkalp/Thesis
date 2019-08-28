import pandas_datareader.data as web
import pandas as pd
import numpy as np

def DetailsofData(data):
    #AlpsFuncs.plotwithslices(data,slices=1,rollingwidth=2,Testoutput = 0,doubleaxis=False)
    #dataret = (data / data.shift(1)) - 1
    #AlpsFuncs.plotwithslices(dataret[1:],slices=20,rollingwidth=4,Testoutput = 0,doubleaxis=False)
    print('\n \n# of   Days in data: %5.0f' %(len(data)))
    print('# of Months in data: %3.1f' %(len(data)/21))
    print('# of  Years in data: %3.1f' %(len(data)/250))

from pmdarima.arima import auto_arima
def custom_arima(data, split = 0.3, m=12):
    print("\n          -- Fitting model with %3.2f split -- " %(split))
    if split < 1:
        testlen = int(len(data)*split)
        trainlen = len(data) - testlen
        train = data[:trainlen]
    if split == 1:
        train = data
    model = auto_arima(train,  m=m, seasonality=True, trace=True,
                               error_action='ignore',  # don't want to know if an order does not work
                               suppress_warnings=True,  # don't want convergence warnings
                               stepwise=True)
    return model


from statsmodels.tsa.stattools import adfuller
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
        
from statsmodels.tsa.stattools import kpss
def kpss_test(data0):
    if isinstance(data0, pd.DataFrame):
        data = data0.iloc[:,0].values
    else:
        data = data0
    print ('Results of KPSS Test:')
    kpsstest = kpss(data, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

def normplot(data):
    plt.hist(data, bins = 40, normed=True)
    mean, std = stats.norm.fit(data)                        # Get mean and standard deviation for Distribution input
    lin = np.linspace(min(plt.xticks()[0]), max(plt.xticks()[0]) , len(data)) # Min and max ticks for computing dist to be plotted
    Gen_dist = stats.norm.pdf(lin, mean, std)               # Generate Distribution
    plt.plot(lin, Gen_dist, label="Norm");plt.show()
    # normalitytest(model.resid()[10:]) 
#normplot(model.resid()[10:])

import matplotlib.pyplot as plt
#from statsmodels.graphics.tsaplots import plot_acf
def plotwithslices(data,figsize=(16,4),slices=5, RollingDistance=20,rollingwidth=1,stdwidth = 1, Testoutput = 0, doubleaxis=True):
    """
    Slice the data and prefirm DickeyFuller and scpi norm test
    """
    print("\n -- Plotting Data with %i Slices and RollingDist of %i and %i-- "  %(slices,RollingDistance,RollingDistance*5))
    totallen=len(data)
    slicelens=int(totallen/slices)
    fig, ax1=plt.subplots(figsize=figsize)
    plt.plot(data,label = 'Data',linewidth = 2)
    plt.plot(data.rolling(RollingDistance).mean(),label = '%i R Mean' %(RollingDistance), linestyle=':', linewidth=rollingwidth)
    plt.plot(data.rolling(RollingDistance*5).mean(),label = '%i R Mean' %(RollingDistance*5), linestyle=':', linewidth=rollingwidth);plt.legend()
    if (doubleaxis==True):
        ax2 = ax1.twinx()
    plt.plot(data.rolling(RollingDistance).std(),color='r',linewidth=stdwidth, label = '%i R Std' %(RollingDistance-1))
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

#google this: v27i03 says:
#For non-seasonal data, we consider ARIMA(p, d, q) models where d is selected 
#based on successive KPSS unit-root tests (Kwiatkowski et al. 1992). That is, we test the data for a unit
#root; if the test result is significant, we test the differenced data for a unit root; and so on.
#We stop this procedure when we obtain our first insignificant result.
def fitandplot(data0, model = 1, split = 0.3, figuretype = 1, returnpredictions = 0, printmodelsummary =1):
    #if split = 1, fits all of the data.
    if isinstance(data0, pd.DataFrame):
        data = data0.iloc[:,0].values
    else:
        data = data0
    #set parameters
    
    testlen = int(len(data)*split)
    trainlen = len(data) - testlen
    train = data[:trainlen]
    test = data[trainlen:]
    orgdatacolor = 'b'; orglinewidth = 1.5
    
    #fit the model and do predictions
    if model ==1:
        print("\n          -- Fitting Model & plotting the forecast -- \n")
        print("Model Not found, fitting:")
        model = custom_arima(train,split=1)
    if printmodelsummary ==1:
        print(model.summary())
    predictions, confint = model.predict(n_periods=testlen, return_conf_int=True)
    in_sample_preds = model.predict_in_sample()
    in_sample_preds[0] = train[0]
    
    print("\n          -- Plotting all data with forecast -- ")
    #plot all the data
    
    plt.figure(figsize=(12,6))
    plt.plot(data,label = 'Test Data',color=orgdatacolor,linewidth=orglinewidth)
    plt.plot(in_sample_preds,linestyle='--',label = 'Preds')
    plt.plot(np.arange(testlen)+len(train), predictions,linestyle='--',label = 'Predictions')
    plt.fill_between(np.arange(testlen)+len(train), confint[:, 0], confint[:, 1], color='k', alpha=.05)
    plt.plot(np.arange(testlen)+len(train), confint,linestyle='--',label = 'ConfInt',linewidth=0.5)
    plt.title('All data with in and out sample predictions with --Split: %2.2f--' %(split)); plt.legend(); plt.grid();plt.show()
    
    if returnpredictions == 0:
        return
    return predictions


def autoarimaR(data):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    import rpy2.robjects.packages as rpackages
    rpackages.importr('forecast')
    autoarima = robjects.r['auto.arima']
    r_dataframe = pandas2ri.py2ri(data)
    modelR = autoarima(r_dataframe)
    return modelR

def montecarloR(data, model=False, modelR = False, simlen = 10,simtimes = 5,plot = False):
    import rpy2.robjects.packages as rpackages
    rpackages.importr('quantmod')
    rpackages.importr('forecast')
    import rpy2.robjects as robjects
    simulate = robjects.r['simulate']
    toString = robjects.r['toString']
    asnumeric = robjects.r['as.numeric']
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    """ Start """
    #simulate(modelR,nsim = simlen)
    if modelR == False:
        modelR = autoarimaR(data)
    if model == False:
        model = auto_arima(data,   seasonal=False, trace=True,
                           error_action='ignore',  # don't want to know if an order does not work
                           suppress_warnings=True,  # don't want convergence warnings
                           stepwise=True)

    orgdataandsim = np.append(data.values[-100:],(model.predict(n_periods=simlen)))
    montecarlo = [np.array(asnumeric(simulate(modelR,nsim = simlen))) for i in range(simtimes)]
    montecarlo = np.transpose(montecarlo)
    modelparam = list(toString(modelR))
    if plot == True:
        plt.plot(range(len(data.values[-100:]),len(montecarlo)+len(data.values[-100:])),montecarlo,linestyle='--',linewidth=1)
        plt.plot((orgdataandsim),linewidth=2,label = 'Org Data')
        plt.title('Fit model %s, MC simlen: %i, simtim: %i' %(modelparam,simlen, simtimes)); plt.legend()
        plt.show()
    return montecarlo , modelparam

def ReturnOutlierArray(data, SigmaS = 3, SigmaE = 20 ,graphs= False):
    cumulativeoutliers = pd.DataFrame(index = data.index)
    for j in range(1,SigmaE):
        sigmamultiplier = j
        figsizes = (8,4)
        AbsRets = abs(data.pct_change())
        Limit = sigmamultiplier * np.std(AbsRets)
        Outliers =  AbsRets > Limit
        Target = pd.Series(0, index = Outliers.index, name='Out' ) 
        for i in range(len(Outliers)): Target[i]=  1 if Outliers[i] else 0
        # Dates only: 
        OutlierDates = AbsRets.loc[AbsRets > Limit].index
        
        if graphs== True:
            # Returns and outliers
            fig, ax1=plt.subplots(figsize=figsizes)
            plt.title('Outlier detection, Returns and Limit %i sigma' %(sigmamultiplier))
            plt.plot(AbsRets,color='k',linestyle='--',linewidth=0.5)
            plt.axhline(y = Limit ,color='r',linestyle='--',linewidth=1, label ='Limit %i sigma' %(sigmamultiplier));plt.grid();plt.legend(loc = 2)
            plt.show()
            # Returns marked on graph
            fig, ax1=plt.subplots(figsize=figsizes)
            plt.title('Outlier detection, Price Chart and Limit %i sigma' %(sigmamultiplier))
            plt.plot(data,linewidth = 1, color='k');plt.legend(loc = 2);plt.legend(loc = 2);plt.grid();ax2 = ax1.twinx()
            from datetime import timedelta
            timerange = timedelta(days=+2)
            for i in range(0,len(OutlierDates)): ax1.axvspan(OutlierDates[i]-timerange, OutlierDates[i]+timerange, alpha=0.5, color='red')
            plt.show()
            print('Number of outliers: ',len(OutlierDates))
            print(OutlierDates)
            
        cumulativeoutliers[j] = Target.values
        
        print('Iteration:%3i,  Limit: %4.4f,  # of Outliers: %3i' %(j,Limit,cumulativeoutliers[j].sum()))
        if cumulativeoutliers[j].sum() == 0: print('\n','-'*30,'\n',' '*5,'No outliers left \n','-'*30); break
    Target = [cumulativeoutliers.iloc[i].sum() for i in range (len(cumulativeoutliers))]
    for i in range(len(Target)): Target[i] = np.nan if Target[i] < SigmaS else Target[i]
    
    if graphs == True:
        figsizes = (16,8)
        fig, ax1=plt.subplots(figsize=figsizes)
        plt.plot(data, color='k');plt.legend();ax2 = ax1.twinx()
        plt.plot(data.index,Target,'s');plt.legend();ax2.set_yticks(np.arange(0, 20));plt.grid();plt.show()

    return Target#, OutlierDates, cumulativeoutliers

#data1 = (abs(data.pct_change()).dropna() / np.std(abs(data.pct_change()))).apply(np.floor)
#data1[2 > data1] = 0

#target= ReturnOutlierArray(data,graphs= False)


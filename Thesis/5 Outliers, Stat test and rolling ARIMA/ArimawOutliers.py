import numpy as np
import pandas as pd
import random

random.seed(42)
import matplotlib.pyplot as plt
import pandas_datareader as web

import seaborn as sns
sns.set()
########## Time Series Example
dataset = pd.DataFrame(web.DataReader("SWED-A.ST", "yahoo", start = "2010")['Adj Close'])
plt.plot(dataset['Adj Close'],label = 'SWED-A.ST', color = 'k', linewidth = 1); plt.legend(); plt.show()

dataset_ret = dataset['Adj Close'].pct_change()
plt.hist(dataset_ret)

from scipy import stats
def normplot(data):
    plt.hist(data, bins = 50, normed=True)
    mean, std = stats.norm.fit(data)                        # Get mean and standard deviation for Distribution input
    lin = np.linspace(min(plt.xticks()[0]), max(plt.xticks()[0]) , len(data)) # Min and max ticks for computing dist to be plotted
    Gen_dist = stats.norm.pdf(lin, mean, std)               # Generate Distribution
    plt.plot(lin, Gen_dist);plt.title('Swedbank Percent Returns');plt.show()
    # normalitytest(model.resid()[10:]) 
normplot(dataset_ret)
stats.probplot(dataset_ret, dist="norm", plot=plt)
plt.show()
sm.qqplot(dataset_ret, line='s')
########## Rolling Variance of Swedbank
plt.plot(dataset,label = 'SWED-A.ST', color = 'k', linewidth = 1); plt.legend(); plt.show()
plt.plot(dataset.rolling(50).std(),label = '%i Day Rolling Variance of SWED-A.ST' %(50),  color ='k', linewidth=1); plt.legend(); plt.show()

########## Stationarity test
def plotwithslices(data,title,figsize=(16,4), slices=5, RollingDistance=20,rollingwidth=1,stdwidth = 1, Testoutput = 0, doubleaxis=True):
    """
    Slice the data and prefirm DickeyFuller and scpi norm test
    """
    print("\n -- Plotting Data with %i Slices and RollingDist of %i and %i-- "  %(slices,RollingDistance,RollingDistance*5))
    totallen=len(data)
    slicelens=int(totallen/slices)
    fig, ax1=plt.subplots()
    plt.plot(data,label = title,color='k',linewidth = 1)
    for i in range(slices-1):    
        plt.axvline(x=data.index[int(totallen/slices)*(i+1)],color='r',linestyle='--',linewidth=1)
    plt.legend();
    plt.show()
    
    if (Testoutput == True):
        print(' Slice  0 S: %4i E: %4i  ' %(0, totallen), end="")
        DickeyFullerPrint(data[0:totallen])
        #normalitytest(data[0:totallen])
        print('\n')
        for i in range(slices):   
            print('\n Slice %2i S: %4i E: %4i  ' %(i+1, slicelens*(i), slicelens*(i+1)), end="")
            DickeyFullerPrint(data[slicelens*(i):slicelens*(i+1)])
            #normalitytest(data[slicelens*(i):slicelens*(i+1)])
    return
plotwithslices(dataset, title = 'SWED-A.ST', Testoutput = 1, doubleaxis = 1)
datasetrets = ((dataset) - (dataset.shift(1))).dropna()
datasetpercrets = dataset.pct_change(1).dropna()
datasetlrets = (np.log(dataset) - np.log(dataset.shift(1))).dropna()
plotwithslices(datasetrets, title = 'SWED-A.ST returns', Testoutput = 1, doubleaxis = 1)
plotwithslices(datasetpercrets, title = 'SWED-A.ST percentage returns', Testoutput = 1, doubleaxis = 1)
plotwithslices(datasetlrets, title = 'SWED-A.ST log returns', Testoutput = 1, doubleaxis = 1)
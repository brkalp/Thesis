import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# getting data
Ticker = "SWED-A.ST"
data = web.DataReader(Ticker, 'yahoo', "2017")['Adj Close']
data.name = Ticker


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
        figsizes = (10,6)
        fig, ax1=plt.subplots(figsize=figsizes)
        plt.plot(data, color='k');plt.legend(loc= 2);ax2 = ax1.twinx()
        plt.plot(data.index,Target,'s', color='r',label = 'Outlier Levels');plt.legend(loc= 1);ax2.set_yticks(np.arange(0, 20));plt.grid();plt.show()

    return Target, OutlierDates, cumulativeoutliers



target, outlierDates, cumulativeoutliers = ReturnOutlierArray(data,graphs= True)

data1 = (abs(data.pct_change()).dropna() / np.std(abs(data.pct_change()))).apply(np.floor)
data1[2 > data1] = 0
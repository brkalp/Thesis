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
SIMSTART = -90; SIMLEN = 50; BREAKTIME = 15; UPPERSIGMA = 12; SIMTIM = 500
from keras.models import load_model
classifier = load_model('my_model.h5')


Test = pd.DataFrame(data)
Test['Returns'] = data.pct_change()
Test['ActPreds'] = abs(Test['Returns'] / - (- data.pct_change().mean() - data.pct_change().std()))
Test['AvgPreds'] = 0
T2 = time.time()
for i in range (0,50)#len(Test)-50):
    start = i
    end = 50 + i
    print("Start:  %4i/%5i "%(start,len(Test)), end ='')
    print("   End: %4i "%end, end ='')
    print("   Time Spent: %4i Finish Rat: %3.3f"%(T2-time.time(), i/len(Test) ))
    Ddata = data[start:end]
    DInput = pd.Series(data[start:end].pct_change().dropna(),index = data[start:end].pct_change().dropna().index)
    Input = np.array( DInput ).reshape(-1,1)
    y_pred2 = pd.Series(classifier.predict(Input.T)[0],index = data[start:end].pct_change().dropna().index)
    ACTpred = pd.Series(-(Input.T / - (- data.pct_change().mean() - data.pct_change().std()))[0],index = data[start:end].pct_change().dropna().index)
    
    fig, ax = plt.subplots(figsize = (15,4))
    plt.subplot(1,3,1);plt.title('Data')
    plt.plot(Ddata)
    plt.plot(Ddata,'.')
    fig.autofmt_xdate()
    plt.subplot(1,3,2);plt.title('input')
    plt.plot(DInput)
    plt.plot(DInput,'.')
    plt.subplot(1,3,3);plt.title('output')
    plt.plot(y_pred2)
    plt.plot(abs(ACTpred),color='r', linestyle = '-',linewidth = 0.5)
    plt.plot(abs(ACTpred),'.')
    plt.tight_layout()
    plt.savefig('Savehere2/%i.jpeg'%i)
#    if i % 30 == 0:
plt.show()
    Test[i] = y_pred2
# Takes around 15 seconds without saving graphs with 2382 datapoints
Test['AvgPreds'] = pd.Series( [Test.iloc[i:i+1,4:].mean(skipna = True,axis = 1).values[0] for i in range(len(Test))], index = Test.index)

#Test.to_excel('Savehere/Test.xlsx')
#Test.to_csv('Savehere/Test.csv')
#Test.iloc[:,0:4].to_csv('Scoring.csv')

print('Entire run time: %i min'%((TT-time.time())/60))
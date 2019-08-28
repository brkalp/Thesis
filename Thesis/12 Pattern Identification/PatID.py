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


import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from datetime import timedelta
sns.set()
# Getting data
Ticker = "SWED-A.ST"
data = web.DataReader(Ticker, 'yahoo', "2010" ,'2019-06-23' )['Adj Close']; 
data = pd.DataFrame(data,index=data.index)
data['Returns'] = data.pct_change(); data = data.dropna()
data.name = Ticker; SIGMULT = 5
LIMIT = data['Returns'].std() * SIGMULT
data['Dev'] =  (data['Returns'] - data['Returns'].mean()) / data['Returns'].std()
#data['DevFloor'] = data['Dev'].apply(np.floor)

split = 95
# Showing the outlier
fig, ax1=plt.subplots()
plt.plot(data['Adj Close'],'k',linewidth = 0.5);plt.title(f'{Ticker} Stock Price')
ax1.axvspan(data.index[-split]-timedelta(days=+5), data.index[-1], alpha=0.2, color='red');plt.show()
# This is where the break is data.index[-84]
#plt.plot(data[:-75]); plt.plot(data)

# ModelR0 is before the Outlier
datatoR0 = data['Adj Close'][:-84]
r_dataframe = pandas2ri.py2ri(datatoR0)
modelR0 = autoarima(r_dataframe)
print(coef(modelR0))

# ModelR1 is After the Outlier
datatoR1 = data['Adj Close']
r_dataframe = pandas2ri.py2ri(datatoR1)
modelR1 = autoarima(r_dataframe)
print(coef(modelR1))

SIMLEN = 30
SIMTIM = 4000 # I will generate 1 with normal rets and 1 with out of sample returns. total is before the division.

montecarlo0 = np.transpose([np.array(asnumeric(simulate(modelR0,nsim = SIMLEN))) for i in range(SIMTIM)])
montecarlo0 = pd.DataFrame( montecarlo0, index=[datatoR0.index[-1] + timedelta(days = + i) for i in range(0, SIMLEN)] )
# Each column is one path
plt.plot( (montecarlo0.iloc[:,:10]) ,'r')
plt.plot( (montecarlo0[0]) ,'r',label = 'First Model')
#plt.plot( (datatoR0[-150:]) )

montecarlo1 = np.transpose([np.array(asnumeric(simulate(modelR1,nsim = SIMLEN))) for i in range(SIMTIM)])
montecarlo1 = pd.DataFrame( montecarlo1, index=[datatoR1.index[-1] + timedelta(days = + i) for i in range(0, SIMLEN)] )
# Each column is one path
plt.plot( (montecarlo1.iloc[:,:10]) ,'b')
plt.plot( (montecarlo1[0]) ,'b',label = 'Second Model')
plt.plot( (datatoR1[-250:]), 'k' ,label = Ticker);plt.legend();plt.title('Monte carlo simulations of identified models');plt.show()

rets0 = pd.DataFrame(montecarlo0.pct_change().dropna().values, columns = np.array([i for i in range(0,len(list(montecarlo0)) ) ]))
#rets1 = pd.DataFrame(montecarlo1.pct_change().dropna().values, columns = np.array([i for i in range(len(list(montecarlo0)),len(list(montecarlo0)) + len(list(montecarlo1)) ) ]))
rets1 = pd.DataFrame(montecarlo1.pct_change().dropna().values, columns = np.array([i for i in range(0,len(list(montecarlo0)) ) ]))


XX = pd.DataFrame()#columns = range(0,SIMTIM * 2), 
YY = pd.Series([i % 2 for i in range (0 , SIMTIM * 2)]) # len(y)
counter = 0
for i in range(0,SIMTIM):
    XX = XX.append(rets0[i])
    XX = XX.append(rets1[i])
XX = XX.T
XX.columns = list( [ i for i in range(SIMTIM*2)])


# Split
split = int(len(list(XX))*0.5)
X_train = XX.iloc[:,:split].T ; X_test = XX.iloc[:,split:].T
y_train = YY.iloc[:split]        ; y_test = YY.iloc[split:]
plt.title('Inputs outputs, in and out of sample')
plt.plot(X_train);plt.plot(X_test);plt.plot(y_train);plt.plot(y_test);plt.show()
#Split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)
#ANN
from keras.models import Sequential
from keras.layers import Dense
def BuildtheNetwork(X_train, y_train):
    classifier = Sequential()
    classifier.add(Dense(units = 128,                       # 128 neurons in the layer
                         kernel_initializer = 'uniform',    # Starting values for the weights in the neurons
                         activation = 'sigmoid',               # Activation function of neurons is Rectified Linear Unit function or ‘relu’.
                         input_dim = X_train.shape[1]))           # Equal to the # of columns of input, Price Rise
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'sigmoid')) # a hidden middle layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # a hidden middle layer
    classifier.compile(optimizer = 'adam',                  # an extension of gradient descent
                       loss = 'mean_squared_error', 
                       metrics = ['accuracy'])              # what to evaulate during optimization
    classifier.fit(X_train, y_train.values, batch_size = 5, epochs = 150)
    return classifier
classifier = BuildtheNetwork(X_train, y_train)
y_pred = classifier.predict(X_train)
y_pred= pd.Series(y_pred[:,0], index = range(len(y_pred[:,0])))
print('\n Number of -Total- MC paths:', SIMTIM)
print(' Lenght of each path:', SIMLEN)
print(' Number of mc paths in the test sample:', len(X_train))
print(' Number of values above 0.5 insample preds:', len(y_pred[y_pred<0.5]))
print(' Number of values lower 0.5 insample preds:', len(y_pred[y_pred>0.5]))
# Finding False positives and negatives
PMASK = y_pred>0.5;NMASK = y_pred<0.5 # Values that predictions found to be positive and negative
FPOS = y_train[PMASK][y_train[PMASK] ==0] # if marked 0 on train set but masked to be positive, it is a False positive.
FNEG = y_train[NMASK][y_train[NMASK] ==1] # if marked 0 on train set but masked to be positive, it is a False positive.
print(' Number of False Positives:',len(FPOS)) 
print(' Number of False Negatives:',len(FNEG)) 
print(' Number of Correct: ',SIMTIM - len(FNEG) - len(FPOS)) 

#fig = plt.figure(figsize=(6, 4))
fig = plt.figure(figsize=(12, 4))
#plt.title('Classification of simulations, in sample')
plt.plot(np.array(y_pred),  color='r',   label = 'Predictions', linewidth = 1.5);plt.legend()
plt.plot(y_train.values,    color='k',   label = 'Training Set', linewidth = 0.8);plt.legend();plt.show()

#####
#####
##### Out of Sample
#####
#####

y_pred2= pd.Series(classifier.predict(X_test)[:,0], index = y_test.index)
#y_pred2= pd.Series(y_pred2[:,0], index = range(len(y_pred2[:,0])))
print('\n Number of -Total- MC paths:', SIMTIM)
print(' Lenght of each path:', SIMLEN)
print('\n Number of mc paths out of sample:', len(X_test))
print(' Number of values above 0.5 insample preds:', len(y_pred2[y_pred2<0.5]))
print(' Number of values lower 0.5 insample preds:', len(y_pred2[y_pred2>0.5]))
# Finding False positives and negatives
PMASK = y_pred2>0.5;NMASK = y_pred2<0.5 # Values that predictions found to be positive and negative
FPOS = y_test.values[PMASK][y_test[PMASK] ==0] # if marked 0 on train set but masked to be positive, it is a False positive.
FNEG = y_test.values[NMASK][y_test[NMASK] ==1] # if marked 0 on train set but masked to be positive, it is a False positive.
print(' Number of False Positives:',len(FPOS)) 
print(' Number of False Negatives:',len(FNEG)) 
print(' Number of Correct: ',SIMTIM - len(FNEG) - len(FPOS)) 

#fig = plt.figure(figsize=(6, 4))
fig = plt.figure(figsize=(12, 4))
plt.title('Classification of simulations')
plt.plot(np.array(y_pred2),  color='r',   label = 'Predictions', linewidth = 1.5);plt.legend()
plt.plot(y_test.values,    color='k',   label = 'Training Set', linewidth = 0.8);plt.legend();plt.show()
##
## Swedbank Scan
##
df = pd.DataFrame ( )
df['Rets'] =datatoR1.T.copy().pct_change().dropna()
df['Price'] = datatoR1.T
df['OutputofANN'] = pd.Series()
for i in range(29,len(df)):
    #plt.title('Stock Price to be Scanned')
    #plt.plot(df['Price'],label = 'Rets')
    #plt.plot(df['Price'],label = 'Total Price data')
    #plt.plot(df['Price'].iloc[i-29:i],label = 'Input') ;plt.legend();plt.show()
    dfscan = sc.transform(np.array(df['Rets'][i-29:i]).reshape(1, -1))
    #plt.plot(dfscan.T);plt.show()
    df['OutputofANN'].iloc[i] = classifier.predict(dfscan)[0][0]

# Classification on Swedbank percentage returns
#Plotting Price
fig, ax1=plt.subplots()
plt.plot(df['Price'][-300:]      , label = 'Price',color = 'black', linewidth = 2)
#Plotting Classification Score output of ANN "
ax2 = ax1.twinx();plt.grid()
plt.plot(df['OutputofANN'][-300:], label = 'Classification Score', alpha=0.5, color='red', linewidth = 0.8);plt.legend(loc = 3);plt.show()

# Classification Scores with rolling Classification Scores
fig, ax1=plt.subplots()
ClassScore = pd.Series(np.zeros(len(df)), index = df.index, name = 'ClassScores')
ClassScore[df['OutputofANN'].rolling(30).mean() > 0.95] = df['OutputofANN'].rolling(30).mean()
plt.plot(ClassScore, label = 'Classification Score > 0.95, Rolling mean of 30', alpha=0.5, color='red', linewidth = 0.8);plt.legend(loc = 3)
# Locating Outliers
# Plotting Classification Score output of ANN
ax2 = ax1.twinx();plt.grid()
outliers = pd.Series(np.zeros(len(df)), index = df.index, name = 'Outliers')
MASK = abs(df['Price'].pct_change()) > df['Price'].pct_change().std()*2
outliers[MASK] = df ['Price'].pct_change() [MASK] / df['Price'].pct_change().std()
plt.plot(outliers);plt.show()

from datetime import timedelta
# FILTERED Classification Scores with rolling Classification Scores FILTEREDfig, ax1=plt.subplots()
plt.plot(ClassScore, label = 'Classification Score > 0.95, Rolling mean of 30', color='k', linewidth = 1);plt.legend(loc = 3)
# Locating Outliers
# Plotting Classification Score output of ANN
ax2 = ax1.twinx();plt.grid()
outliers = pd.Series(np.zeros(len(df)), index = df.index, name = 'Outliers')
MASK = abs(df['Price'].pct_change()) > df['Price'].pct_change().std()*2
outliers[MASK] = df ['Price'].pct_change() [MASK] / df['Price'].pct_change().std()
for i in outliers[outliers>0].index: ax1.axvspan(i-timedelta(days=+50), i+timedelta(days=+50), alpha=0.08, color='red')
plt.title('Detected Outliers and actual outliers');plt.show()
#plt.plot(outliers)
# plt.scatter(outliers, ClassScore) This means nothing


# Plot absolute cumulative sum of outliers and Cumulative sum of ANN Scores
fig, ax1=plt.subplots()
for i in outliers[outliers>0].index: ax1.axvspan(i-timedelta(days=+50), i+timedelta(days=+50), alpha=0.04, color='red')
# CumSums of classScores and Outliers
L =plt.plot((ClassScore).cumsum(), label = 'Cumulative Sum of ANN output', linewidth = 1, color = 'k'); ax2 = ax1.twinx();plt.grid()
plt.plot(abs(outliers).cumsum(), label = 'Cumulative Sum of Outlier Levels', linewidth = 1.5, color = 'b')
L = plt.legend();plt.title('Cumulative sums: Detected outliers and Outlier levels')
from matplotlib.lines import Line2D
plt.legend([Line2D([0], [0], color='k', lw=1),Line2D([0], [0], color='b', lw=1)], ['Cumulative Sum of ANN output', 'Cumulative Sum of Outlier Levels'])
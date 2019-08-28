import numpy as np
import pandas as pd
import random
random.seed(42)
import matplotlib.pyplot as plt
import pandas_datareader as web

startdate= "2010"; enddate='2019-06-23'
ticker= "SWED-A.ST"
data0 = web.DataReader(ticker, 'yahoo', start = startdate, end=enddate)
data0['Rets']  = data0['Adj Close'].pct_change()
#
# Data processing
#
df = pd.DataFrame()
df['High']  = data0['High']
df['Low']   = data0['Low']
df['Open']  = data0['Open']
df['Close']   = data0['Close']
df['H-L']     = data0['High']  - data0['Low']
df['O-C']     = data0['Close'] - data0['Open']
df['Adj Close'] = data0['Adj Close']
df['Vol']       = data0['Volume']
df['Rets']       = data0['Rets']
#df['5MA']       = data0['Adj Close'].rolling(window = 5).mean()
#df['10MA']      = data0['Adj Close'].rolling(window = 10).mean()
#df['30MA']      = data0['Adj Close'].rolling(window = 30).mean()
#df['Std_dev5dayroll']= data0['Close'].rolling(5).std()
df = df.dropna()
# This one is the target coloumn
MEAN = data0['Rets'].mean(); STDDEV = data0['Rets'].std()
df['Target'] = (data0['Rets'] - MEAN )/ STDDEV # SigmaLvl
df['Target'][df['Target'].between(-2,2)] = 0
PosOutlierDates = df['Target'][df['Target']>2].index

# Columns that should be shifted 5 days!
COLUMNSTOSHIFT = ['H-L', 'O-C', 'Rets', 'Adj Close', 'Vol']


X = df[COLUMNSTOSHIFT]
#Splitting and Plot
fig, ax1=plt.subplots(figsize=(9,5))
split = int(len(df)*0.90)
plt.title("Range of split and test areas")
X_train = X[:split]; X_test = X[split:]
y_train = df.iloc[:, -1][:split]; y_test = df.iloc[:, -1][split:]
plt.plot(X_train['Adj Close'],label='Train');plt.plot(X_test['Adj Close'],label='Test');plt.grid();plt.legend()
from datetime import timedelta
for i in range(0,len(PosOutlierDates)): ax1.axvspan(PosOutlierDates[i]-timedelta(days=+2), PosOutlierDates[i]+timedelta(days=+2), alpha=0.5, color='red')
#ax2 = ax1.twinx();plt.plot(y)
plt.show()
print('\n Last day of split: ',X_train[-1:].index.format(formatter=lambda x: x.strftime('%Y   %m   %d')))

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test) # Both train and training set means are 0

####
#### X needs to be transformed!
####

df2 = pd.DataFrame(index = df.index)
for COLUMN in COLUMNSTOSHIFT:
    for i in range(0,5):
        df2[f'{-i} {COLUMN}'] = df[f'{COLUMN}'].shift(i)

df2['5MA']       = data0['Adj Close'].rolling(window = 5).mean()
df2['10MA']      = data0['Adj Close'].rolling(window = 10).mean()
df2['30MA']      = data0['Adj Close'].rolling(window = 30).mean()
df2['Std_dev5dayroll']= data0['Close'].rolling(5).std()

df2['Target'] = df['Target'].shift(-1)
df2 = df2.dropna()
X = df2.iloc[:,:-1]
split = int(len(df2)*0.90)
X_train = X[:split]; X_test = X[split:]
y_train = df2.iloc[:, -1][:split]; y_test = df2.iloc[:, -1][split:]

#
# Building the network
#

ACTFUNC = 'relu'
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 128,                       # 128 neurons in the layer
                     kernel_initializer = 'uniform',    # Starting values for the weights in the neurons
                     activation = ACTFUNC,               # Activation function of neurons is Rectified Linear Unit function or ‘relu’.
                     input_dim = X.shape[1]))           # Equal to the # of columns of input, Price Rise
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
#classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
classifier.compile(optimizer = 'adam',                  # an extension of gradient descent
                   loss = 'mean_squared_error', 
                   metrics = ['accuracy'])              # what to evaulate during optimization
classifier.fit(X_train[:-1], y_train[1:].values, batch_size = 10, epochs = 200)

#y_pred = classifier.predict(X_test)
y_pred = classifier.predict(X_train)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
axes[0].plot(y_train.values,alpha=0.7, label = 'Training set', color = 'k')
axes[0].set_title('In sample')
#plt.plot(y_pred,alpha=0.5, label = 'Outcome'); plt.legend(); plt.show()
axes[0].plot(y_pred > 0.5, label = 'Outcome')
axes[0].set_ylabel('Magnitute of outliers')
#Out of sample
y_test2 = classifier.predict(X_test)
axes[1].plot(y_test.values,alpha=0.7, label = 'Test set', color = 'k'); 
axes[1].set_title('Out of sample')
axes[1].plot(y_test2 , label = 'Outcome')
axes[1].legend(loc = 4) 
plt.show()
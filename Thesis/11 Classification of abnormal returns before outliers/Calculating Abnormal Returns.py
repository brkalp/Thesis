# -*- coding: utf-8 -*-

ticker= "SWED-A.ST"
INDEX = "^OMX"
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
startdate= "2010"; enddate='2019-06-23'
df0 = pd.DataFrame()
df0['Stock'] = web.DataReader(ticker, 'yahoo', start = startdate, end=enddate)['Adj Close'];
df0['Index'] = web.DataReader(INDEX, 'yahoo', start = startdate, end=enddate)['Adj Close'];
df0['StRet'] = df0['Stock'].pct_change()
df0['InRet'] = df0['Index'].pct_change()
df0 = df0.dropna()

# Stable Beta model
from statsmodels import regression
import statsmodels.api as sm
def linreg(market,stock):
    market = sm.add_constant(market)
    model = regression.linear_model.OLS(stock,market).fit()
    return model.params[0], model.params[1],model
ALPHA, BETA, model = linreg(df0['InRet'],df0['StRet'])

# Generating Returns created by the market
df0['MarketGenRets'] = df0['InRet']*BETA + ALPHA

# Calculating Abnormal Prices from Abnormal Returns
df0['ABNrets'] = df0['StRet'] - df0['MarketGenRets']

#
# Data processing
#
df = pd.DataFrame()
df['StRet'] = df0['StRet']
df['InRet'] = df0['InRet']
df['Target'] = df0['ABNrets']
df = df.dropna()

COLUMNSTOINPUT = ['StRet','InRet']
X = df[COLUMNSTOINPUT]

#Splitting and Plot
fig, ax1=plt.subplots(figsize=(9,5))
split = int(len(df)*0.90)
plt.title("Range of split and test areas")
X_train = X[:split]
X_test = X[split:]
y_train = df.iloc[:, -1][:split]
y_test = df.iloc[:, -1][split:]
plt.plot(X_train,label='Train');plt.plot(X_test,label='Test');plt.grid();plt.legend()

#
# Building the network
#

ACTFUNC = 'tanh'
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 128,                       # 128 neurons in the layer
                     kernel_initializer = 'uniform',    # Starting values for the weights in the neurons
                     activation = ACTFUNC,               # Activation function of neurons is Rectified Linear Unit function or ‘relu’.
                     input_dim = X.shape[1]))           # Equal to the # of columns of input, Price Rise
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
#classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
classifier.compile(optimizer = 'adam',                  # an extension of gradient descent
                   loss = 'mean_squared_error', 
                   metrics = ['accuracy'])              # what to evaulate during optimization
classifier.fit(X_train, y_train.values, batch_size = 10, epochs = 100)

#y_pred = classifier.predict(X_test)
y_pred = classifier.predict(X_train)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
axes[0].plot(y_train.values,alpha=0.7, label = 'Training set', color = 'k')
axes[0].set_title('In sample')
#plt.plot(y_pred,alpha=0.5, label = 'Outcome'); plt.legend(); plt.show()
axes[0].plot(y_pred, label = 'Outcome')
axes[0].set_ylabel('Abnormal Returns')
#Out of sample
y_test2 = classifier.predict(X_test)
axes[1].plot(y_test.values,alpha=0.7, label = 'Test set', color = 'k'); 
axes[1].set_title('Out of sample')
axes[1].plot(y_test2 , label = 'Outcome')
axes[1].legend(loc = 4) 
plt.show()
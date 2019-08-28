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
SIMSTART = -90; SIMLEN = 50; BREAKTIME = 15; UPPERSIGMA = 12; SIMTIM = 500

Cumulativerets = pd.read_csv('Cumulativerets.csv').set_index('Unnamed: 0')
XX = Cumulativerets.copy()
Y =  (Cumulativerets / - (- data.pct_change().mean() - data.pct_change().std()))
#Y [ (-2 < Y)  & (Y < +2)] = 0
#Y[ Y >  2 ] =  Y.apply(np.floor)
#Y[ Y < -2 ] =  Y.apply(np.ceil)
#plt.plot(Y);plt.show()

XX = XX.T
Y = abs(Y.T)

##Split
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#XX = sc.fit_transform(XX)#; X_test = sc.transform(X_test)
#ANN
ACTFUNC = 'relu'
from keras.models import Sequential
from keras.layers import Dense
def BuildtheNetwork(X_train, y_train):
    classifier = Sequential()
    classifier.add(Dense(units = 128,                       # 128 neurons in the layer
                         kernel_initializer = 'uniform',    # Starting values for the weights in the neurons
                         activation = ACTFUNC,               # Activation function of neurons is Rectified Linear Unit function or ‘relu’.
                         input_dim = X_train.shape[1]))           # Equal to the # of columns of input, Price Rise
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
    classifier.add(Dense(units = len(y_train.T), kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
    classifier.compile(optimizer = 'adam',                  # an extension of gradient descent
                       loss = 'mean_squared_error', 
                       metrics = ['accuracy'])              # what to evaulate during optimization
    classifier.fit(X_train, y_train.values, batch_size = 5, epochs = 200)
    return classifier
classifier = BuildtheNetwork(XX, Y)
y_pred = classifier.predict(XX)

def plotANNaccuracy():
    plt.plot(XX.T['10']);plt.title('INPUT, returns');plt.show()
    #plt.plot(y_pred.T);plt.title('Outcome, ANN output');plt.show()
    plt.plot(Y.T);plt.title('Target');plt.show()
    Output = pd.DataFrame(y_pred.T, index = Y.T.index)
    plt.plot(Output);plt.title('Outcome, ANN output');plt.show()
#plotANNaccuracy()
#classifier.save('my_model.h5')
    
    
plt.plot(range(49),XX.T['10.18'],color = 'k', label = 'Distruption of 10')
plt.plot(range(49),XX.T['5.35'], label = 'Distruption of 5');plt.title('INPUT, returns')
plt.legend();plt.show()
#plt.plot(y_pred.T);plt.title('Outcome, ANN output');plt.show()
plt.plot(range(49),Y.T['10.18'],color = 'k', label = 'Distruption of 10')
plt.plot(range(49),Y.T['5.35'], label = 'Distruption of 5');plt.title('Output, Training Target')
plt.legends();plt.show()
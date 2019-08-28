import numpy as np
import pandas as pd
import random
random.seed(42)
import matplotlib.pyplot as plt
import pandas_datareader as web

# Read data from cleaner
from datetime import datetime
MA = pd.read_csv(r'Cleaner.csv')
COMPNAMES = MA.columns
COMPS = range(len(MA.columns)); LEN = range(len(MA.iloc[:,0])); TOTALLEN = len(MA.columns) * len(MA.iloc[:,0])
COMPNAMES0 = pd.Series(COMPNAMES[int(i/len(LEN))] for i in range(TOTALLEN) )
for II in  COMPS :
    for JJ in  LEN :
        try: MA.iloc[ JJ,II ] = datetime.strptime(MA.iloc[ JJ,II ], '%m %d %Y')
        except: pass
Mergerdates = pd.DataFrame ( [MA.iloc[j,i] for i in COMPS for j in LEN] , index = range(TOTALLEN))
Mergerdates['Acquirer'] = COMPNAMES0
Mergerdates = Mergerdates.dropna().set_index(0)

# Getting price Data
START= "2009"; END='2019-06-23'
df = pd.DataFrame()
df['INDEX'] = web.DataReader('SPY', 'yahoo', start = START, end=END)['Adj Close']
df[COMPNAMES] = web.DataReader(COMPNAMES, 'yahoo', start = START, end=END)['Adj Close']
df.plot(figsize=(8,20),subplots = True)
plt.show()
df = df.pct_change()
#df.to_csv('IndexandTechprices.csv')

# Shifting Mergers that happend in the weekends
from datetime import timedelta
for II in range (0,5):
    Mergerdates['Dates'] = Mergerdates.index
    Mergerdates['IndexPrice'] = df['INDEX']
    Mergerdates['Isna'] = np.isnan(Mergerdates['IndexPrice'])
    print('Number of Mergers outside Index Dates: %3i' %(len( Mergerdates.loc[Mergerdates['Isna']] )))
    Mergerdates.loc[Mergerdates['Isna'],'Dates'] = Mergerdates.loc[Mergerdates['Isna'],'Dates'] + timedelta(days = 1)
    Mergerdates = Mergerdates.set_index('Dates',drop=False)

# Getting Rolling Betas
from scipy import stats
def LINREG(DATA,INDEX):  BETA,ALPHA,__,__,__  = stats.linregress(INDEX,DATA); return ALPHA,BETA
def rollingbeta  (DATA,INDEX,DAYS) : return pd.Series([ LINREG(DATA.iloc[i:DAYS+i], INDEX.iloc[i:DAYS+i])[1] for i in range(len(DATA) - DAYS) ], index = DATA[DAYS:].index)
def rollingalpha (DATA,INDEX,DAYS) : return pd.Series([ LINREG(DATA.iloc[i:DAYS+i], INDEX.iloc[i:DAYS+i])[0] for i in range(len(DATA) - DAYS) ], index = DATA[DAYS:].index)
DAYS = 60
for COMP in COMPNAMES:
    df[f'{COMP} Alpha']  = rollingalpha  (df[f'{COMP}'],df['INDEX'],DAYS)
    df[f'{COMP}  Beta']  = rollingbeta   (df[f'{COMP}'],df['INDEX'],DAYS)
COMPSBETAS = list([f'{COMP}  Beta' for COMP in COMPNAMES])
df[COMPSBETAS].plot(figsize=(8,20),subplots = True);plt.show()

# Calculating Abnormal returns
for COMP in COMPNAMES: df[f'ABNRET {COMP}'] = df[f'{COMP}'] - (df[f'{COMP} Alpha'] + df['INDEX'] * df[f'{COMP}  Beta'])
COMPSABNRETS = list([f'ABNRET {COMP}' for COMP in COMPNAMES])
df[COMPSABNRETS].plot(figsize=(8,20),subplots = True);plt.show()
for COMP in COMPNAMES: print('%4.4s, MAE %3.5f'  %(COMP, abs(df[f'ABNRET {COMP}']).sum()/len(df)) )

POSSHIFT =5 ;NEGSHIFT = 10
# Getting shifted Abn Rets
df2 = pd.DataFrame()
for COMP in COMPSABNRETS:
    for SHIFT in range(-POSSHIFT,NEGSHIFT): df2[f'{-SHIFT} {COMP}'] = df[f'{COMP}'].shift(SHIFT)

# Getting Returns around Mergers
Mergers = pd.DataFrame(Mergerdates['Acquirer'])
for SHIFT in range (-POSSHIFT, NEGSHIFT): Mergers [f'{-SHIFT}'] = 0
for COMP in COMPNAMES:
    for SHIFT in range(-POSSHIFT, NEGSHIFT):
        Mergers[f'{-SHIFT}'][Mergers['Acquirer'] == COMP] = df2.loc[Mergers['Acquirer'][Mergers['Acquirer'] == COMP].index,[f'{-SHIFT} ABNRET {COMP}']].values
Mergers = Mergers.dropna()

# Printing number of mergers for each company
for COMP in COMPNAMES:
    print('%4s ACQ amount %3i ' %(COMP, len(Mergers['Acquirer'][Mergers['Acquirer']==COMP])) )

plt.boxplot(Mergers.iloc[:,1:].values,showfliers=False)

# Entire dataset of returns and Mergers
df3 = pd.DataFrame(index = df.index)
for COMP in COMPNAMES:
    df3[f'MERGER {COMP}'] = 0
    df3.loc[Mergers['Acquirer'][Mergers['Acquirer'] == f'{COMP}' ].index,f'MERGER {COMP}'] = Mergers['Acquirer'][Mergers['Acquirer'] == f'{COMP}' ]
    for SHIFT in range(-POSSHIFT, NEGSHIFT):
        df3[f'{-SHIFT} ABNRET {COMP}'] = df[f'ABNRET {COMP}'].shift(SHIFT)

# Preparing for Neural Network - need a single matrix with limited columns
MATRIXSIZE = int(len(df3.columns) / len(COMPNAMES))
df4 = pd.DataFrame(columns = Mergers.columns, index = range( len(COMPNAMES) * len(df3)) )
for II in range (len(COMPNAMES)):
#    print(II*MATRIXSIZE, (II+1)*MATRIXSIZE) # Column of Target
#    print(II*len(df3), (II+1)*len(df3))     # Row    of New df
    df4.iloc[II*len(df3):(II+1)*len(df3),:] = df3.iloc[:,II*MATRIXSIZE: (II+1)*MATRIXSIZE].values
df4 = df4.dropna()
df4['Target'] = 1
df4['Target'][df4['Acquirer'] == 0] = 0


X = df4[df4.columns[1:-1]]
Y = df4[df4.columns[-1]]
#Splitting and Plot
split = int(len(df4)*0.90)
X_train = X[:split]
X_test  = X[split:]
y_train = Y[:split]
y_test  = Y[split:]
fig, ax1=plt.subplots(figsize=(9,5))
plt.title("Range of split and test areas")
plt.plot(X_train,label='Train')
plt.plot(X_test,label='Test');plt.show()
plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test');plt.show()

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
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
#classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = ACTFUNC)) # a hidden middle layer
classifier.compile(optimizer = 'adam',                  # an extension of gradient descent
                   loss = 'mean_squared_error', 
                   metrics = ['accuracy'])              # what to evaulate during optimization
classifier.fit(X_train, y_train.values, batch_size = 10, epochs = 20)

#y_pred = classifier.predict(X_test)
y_pred = classifier.predict(X_train)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
axes[0].plot(y_pred,linewidth =3, label = 'Outcome')
axes[0].plot(y_train.values,alpha=0.5,linewidth =1, label = 'Training set', color = 'k')
axes[0].set_title('In sample')
#plt.plot(y_pred,alpha=0.5, label = 'Outcome'); plt.legend(); plt.show()
axes[0].set_ylabel('Binary for Mergers')
axes[0].legend(loc = 5) 
#Out of sample
y_test2 = classifier.predict(X_test)
axes[1].plot(y_test.values,alpha=0.5, label = 'Test set', color = 'k'); 
axes[1].set_title('Out of sample')
axes[1].plot(y_test2 , label = 'Outcome')
axes[1].legend(loc = 5) 
plt.show()
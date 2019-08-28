import pandas as pd
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as web

# get adjusted close prices from Yahoo
TICKERS = ['SWED-A.ST', '^OMX']
TICKER = ['SWED-A.ST']
STARTDATE= "2010"; ENDDATE='2019-06-23'
DATA0 = web.DataReader(TICKERS, 'yahoo', start = STARTDATE, end=ENDDATE)['Adj Close']
df = pd.DataFrame()
df['INDEX'] = DATA0[TICKERS[1]]
df['DATA']  = DATA0[TICKERS[0]]
df = df.pct_change().dropna()

import statsmodels.api as sm
def KFLINREG(INDEX, SERIES,DELTA):
    OBSMAT = sm.add_constant(INDEX, prepend=False)[:, np.newaxis]
    TMAT = DELTA / (1 - DELTA) * np.eye(2)
    kf = KalmanFilter(n_dim_state=2, n_dim_obs=1, 
                      initial_state_covariance = np.ones((2, 2)), # Starting point for random matrices
                      initial_state_mean       = np.array([1,0]), # Starting for Beta and alpha
                      transition_matrices      = np.eye(2),       # coefs for beta states
                      transition_covariance    = TMAT,            # coef for random of beta
                      observation_matrices     = OBSMAT,          # Index Vector with: ones vector for alphas
                      observation_covariance   = 1)               # coef for random of index
    return kf.filter(SERIES)
STATES, __ = KFLINREG(df['INDEX'].values, df['DATA'].values, DELTA = 0.1) # __ is for coefs of error terms
KFESTM = pd.DataFrame({'Beta': STATES[:, 0], 'Alpha':STATES[:, 1]}, index=df.index)

from scipy import stats
def LINREG(DATA,INDEX):  BETA,ALPHA,__,__,__  = stats.linregress(INDEX,DATA); return ALPHA,BETA
def rollingbeta  (DATA,INDEX,DAYS) : return pd.Series([ LINREG(DATA.iloc[i:DAYS+i], INDEX.iloc[i:DAYS+i])[1] for i in range(len(DATA) - DAYS) ], index = DATA[DAYS:].index)
def rollingalpha (DATA,INDEX,DAYS) : return pd.Series([ LINREG(DATA.iloc[i:DAYS+i], INDEX.iloc[i:DAYS+i])[0] for i in range(len(DATA) - DAYS) ], index = DATA[DAYS:].index)

DAYS = 30
df['Alpha'] = rollingalpha  (df['DATA'],df['INDEX'],DAYS)
df['Beta']  = rollingbeta   (df['DATA'],df['INDEX'],DAYS)
df['KF Alpha'] = KFESTM['Alpha'].iloc[DAYS:]
df['KF Beta']  = KFESTM['Beta'].iloc[DAYS:]
df = df.dropna()
df['OLS Beta'] = df['Beta']
df[['OLS Beta','KF Beta']].plot(subplots=True)
plt.show()

df['OLS Abnormal Returns'] = df['DATA'] - (df['Beta']   *df['INDEX'] + df['Alpha'])
df['KF  Abnormal Returns'] = df['DATA'] - (df['KF Beta']*df['INDEX'] + df['KF Alpha'])
DATA0[[TICKERS[0],TICKERS[1]]].plot(subplots=True)
plt.show()
df[['OLS Abnormal Returns','KF  Abnormal Returns']].plot(subplots=True)
plt.show()
# Print errors
print('MAE of OLS %3.7f' %(abs(df['OLS Abnormal Returns']).sum()/len(df['OLS Abnormal Returns'])))
print('MAE Ret of  KF %3.7f' %(abs(df['KF  Abnormal Returns']).sum()/len(df['KF  Abnormal Returns'])))
## Norm test
#from scipy import stats
#__, p = stats.normaltest(df['OLS Abnormal Returns']);print("p = {:g}".format(p)) # not normal
#__, p = stats.normaltest(df['KF  Abnormal Returns']);print("p = {:g}".format(p)) # less normal
#print("Mean of OLS %3.9f"%df['OLS Abnormal Returns'].mean() ) # not normal
#print("Mean of KF  %3.9f"%df['KF  Abnormal Returns'].mean() ) # less normal

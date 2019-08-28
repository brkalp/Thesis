import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(web.DataReader("SWED-A.ST", "yahoo", start = "2010")['Adj Close'])
plt.plot(df['Adj Close'],label = 'SWED-A.ST', color = 'k', linewidth = 1); plt.legend(); plt.show()
from pmdarima.arima import auto_arima
model = auto_arima(df,  seasonality=False, trace=True,
                           error_action='ignore',  # don't want to know if an order does not work
                           suppress_warnings=True,  # don't want convergence warnings
                           stepwise=True)
df['Returns'] = df['Adj Close'].pct_change()
df['insamplepreds'] = model.predict_in_sample()
df['Errors'] = pd.DataFrame(model.resid(),columns=['Errors'],index=df.index)[1:]
df['RMS'] = pd.DataFrame(np.power(model.resid(),2),columns=['RootMeanSquared'],index=df.index)[1:]
df['MAPE'] = pd.DataFrame(abs(model.resid())/df['Adj Close'])[1:]

fig, ax1=plt.subplots()
plt.plot(df['Adj Close'], 'k', label='Data');plt.legend(loc=1)
ax2 = ax1.twinx()
plt.plot(df['RMS'],color='b',linewidth=1)
plt.legend(loc = 1)
df['Sigma Deviation'] = abs( (df['Returns'] - np.mean(df['Returns'])) / np.std(df['Returns']) )
df['Sigma Deviation2'] = df['Sigma Deviation'].apply(np.floor)
len(df['Sigma Deviation2'][df['Sigma Deviation2']>0]) #500
from datetime import timedelta
for i in df['Sigma Deviation2'][df['Sigma Deviation2']>4].index: ax1.axvspan(i-timedelta(days=+10), i+timedelta(days=+10), alpha=0.2, color='red')
plt.title('ARIMA model residuals with error terms');plt.show()

model.predictions()

SLICE = 2200
df2 = df[:SLICE].copy()
model = auto_arima(df2['Adj Close'],  seasonality=False, trace=True,
                           error_action='ignore',  # don't want to know if an order does not work
                           suppress_warnings=True,  # don't want convergence warnings
                           stepwise=True)
df2['RMS'] = pd.DataFrame(np.power(model.resid(),2),columns=['RootMeanSquared'],index=df2.index)[1:]
Predictions = pd.DataFrame(model.predict(n_periods=2382-SLICE, return_conf_int=True)[0], columns=['Predictions'],    index = df[SLICE:].index)
confint = pd.DataFrame(model.predict(n_periods=2382-SLICE, return_conf_int=True)[1], columns=['Lower','Upper'],    index = df[SLICE:].index)
df3 = pd.DataFrame(df['Adj Close'][SLICE:].copy())
df3['Predictions'] = Predictions
df3['Error'] = np.power(df3['Predictions'] - df3['Adj Close'],2)

fig, ax1=plt.subplots()
plt.plot(Predictions,color='b',linewidth=1,label='Predictions')
plt.plot(df2['Adj Close'],color='k',linewidth=1,label='Prices')
plt.plot(df3['Adj Close'],color='k',linewidth=1)
plt.fill_between(confint.index,confint['Lower'],confint['Upper'], color='k', alpha=.15);plt.legend(loc = 2)
ax2 = ax1.twinx()
plt.plot(df2['RMS'],color='r',linewidth=1,linestyle=':')
plt.plot(df3['Error'],color='r',linewidth=1,linestyle=':',label='');plt.legend(loc = 6);plt.title('ARIMA model with out of sample errors')
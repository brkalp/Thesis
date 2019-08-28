# https://rpy2.readthedocs.io/en/version_2.8.x/introduction.html
import pandas_datareader.data as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import datetime
from datetime import timedelta
sns.set()
# Getting Data
TICKER = "SWED-A.ST"; INDEX = "^OMX"
data0 = pd.DataFrame(web.DataReader(TICKER,     "yahoo","2007","2019")['Adj Close'])
index = pd.DataFrame(web.DataReader(INDEX,      "yahoo","2007","2019")['Adj Close'])
data0['Rets'] = data0.pct_change();index['Rets'] = index.pct_change()
index = index.dropna(); data0 = data0.dropna()


# Plotting
fig, ax1 = plt.subplots()
LNS1 = ax1.plot(data0['Adj Close'], 'k', label = TICKER)
ax1.set_xlabel('Dates')
ax1.set_ylabel('%s Price in \$' %TICKER)
ax2 = ax1.twinx()
LNS2 = ax2.plot(index['Adj Close'], label = INDEX)
ax2.tick_params(axis='y')
ax2.set_ylabel('%s Price in \$' %INDEX)
fig.tight_layout();LNS = LNS1 + LNS2
LABS = [l.get_label() for l in LNS]
ax2.legend(LNS, LABS, loc=0);plt.grid();plt.show()

# Getting Betas
from scipy import stats
def getbeta(index,rets):  BETA  = stats.linregress(index,rets)[0]; return BETA
def getalpha(index,rets): ALPHA = stats.linregress(index,rets)[1]; return ALPHA
df = data0.copy()
df['IndexPri'] = index['Adj Close']
df['IndexRet'] = index['Rets']
df = df.dropna()
RANGES = [5,15,30,60,90,180,360,720]
#RANGES = [30,90,1800]#
for ROLLINGRANGE in RANGES:
    df[f'{ROLLINGRANGE}DAlpha'] = pd.Series([ getalpha(df['Rets'].iloc[i:ROLLINGRANGE+i], df['IndexRet'].iloc[i:ROLLINGRANGE+i]) for i in range(len(data0) - ROLLINGRANGE) ], index = data0[ROLLINGRANGE:].index)
    df[f'{ROLLINGRANGE}DBeta']  = pd.Series([ getbeta(df['Rets'].iloc[i:ROLLINGRANGE+i], df['IndexRet'].iloc[i:ROLLINGRANGE+i]) for i in range(len(data0) - ROLLINGRANGE) ], index = data0[ROLLINGRANGE:].index)

# Calculating Expected Returns and Abnormal Returns
for RANGE in RANGES:
    df[f'{RANGE}D ExpRet'] = df[f'{RANGE}DBeta']*df['IndexRet'] + df[f'{RANGE}DAlpha']
    df[f'{RANGE}D AbnRet'] = df['Rets'] - df[f'{RANGE}D ExpRet']

data0 = data0['2010-1-1':].copy()
data0['2010-1-1':].copy()

#Finding outliers after 2010
STDDEV = data0['Rets'].std(); MEAN = data0['Rets'].mean()
data0['SigmaLvl'] = (data0['Rets'] - MEAN) / STDDEV
data0['SigmaLvl'][data0['SigmaLvl'].between(-2,2)] = 0
print('Number of positive outliers: %i' %(len(data0['SigmaLvl'][data0['SigmaLvl']>0])) )
print('Number of Negative outliers: %i' %(len(data0['SigmaLvl'][data0['SigmaLvl']<0])) )
dfP = pd.DataFrame(data0['SigmaLvl'][data0['SigmaLvl']>0]) # Dataframe for positive outliers
dfN = pd.DataFrame(data0['SigmaLvl'][data0['SigmaLvl']<0]) # Dataframe for Negative outliers

# Plot setups
COLS   = [str(RANGE) + 'D' for RANGE in RANGES]
LABELS = [str(RANGE) + ' Days' for RANGE in RANGES]
ALPHA = np.ones(len(COLS))
PLOTSTART = -2300
PLOTEND = -2000
DATESTART = pd.to_datetime(df.iloc[PLOTSTART:PLOTSTART+2].index)[0]
DATEEND = pd.to_datetime(df.iloc[PLOTEND:PLOTEND+2].index)[0]
def plotoutliers(DATESTART):
#     dfP = dfP.loc[DATESTART:,:]; dfN = dfN.loc[DATESTART:,:]
    TIMETOSHADE = 1
    for i in dfP['SigmaLvl'].index: 
        if i > DATESTART:
            if i < DATEEND: plt.axvspan(i-timedelta(days=+TIMETOSHADE), i+timedelta(days=+TIMETOSHADE), alpha=0.1, color='blue') 
    for i in dfN['SigmaLvl'].index: 
        if i > DATESTART: 
            if i < DATEEND:  plt.axvspan(i-timedelta(days=+TIMETOSHADE), i+timedelta(days=+TIMETOSHADE), alpha=0.1, color='red')

# Plotting Betas
def plotthese(VARIABLE):
    fig, ax = plt.subplots()
    for counter in range( len(COLS) ) :
        plt.plot(df[f'{COLS[counter]}{VARIABLE}'].iloc[PLOTSTART:PLOTEND], linewidth = 1, label = LABELS[counter], alpha = ALPHA[counter])
    plotoutliers(DATESTART)
    plt.title(f'{VARIABLE} estimations with different ranges')
    if VARIABLE == ' AbnRet':
        plt.title('Abnormal Returns ($Y_i - \hat{Y}_i$) with different \u03B2 estimations')
    fig.autofmt_xdate()
    plt.legend()
    plt.show()

plt.plot(df['Rets'].iloc[PLOTSTART:PLOTEND])
plotthese('Beta')
plotthese('Alpha')
plotthese(' AbnRet')

# Printing Data, MAYBE USE MAPE?
for counter in range( len(COLS) ) :
    LASTDAYS = -2000
    print('Last %5i days: %10s Rolling Beta, Abn. Return \u03bc is %8.5f, MAE: %5.5f Avg \u03B1: %4.4f, Avg \u03B2: %4.4f ' 
          %(LASTDAYS,f' {RANGES[counter]} Days', 
                          df[f'{COLS[counter]} AbnRet'].iloc[LASTDAYS:].mean(), 
                          abs(df[f'{COLS[counter]} AbnRet'].iloc[LASTDAYS:]).sum()/-LASTDAYS,
                          df[f'{COLS[counter]}Alpha'].iloc[LASTDAYS:].mean(), 
                          df[f'{COLS[counter]}Beta'].iloc[LASTDAYS:].mean() ))

dfERRcheck = abs(df.iloc[-2000:]).copy()

## Printing Data, MAYBE USE MAPE?
#for counter in range( len(COLS) ) :
#    LASTDAYS = -2000
#    print('Last %5i days: %10s Rolling Beta, MAE: %5.5f Avg \u03B1: %4.4f, Avg \u03B2: %4.4f ' 
#          %(LASTDAYS,f' {RANGES[counter]} Days', 
#                          abs(df[f'{COLS[counter]} AbnRet'].iloc[LASTDAYS:]).sum()/-LASTDAYS,
#                          df[f'{COLS[counter]}Alpha'].iloc[LASTDAYS:].mean(), 
#                          df[f'{COLS[counter]}Beta'].iloc[LASTDAYS:].mean() ))

#PlotBoxplot of Returns before outliers
dfRET = data0['2010-1-1':].copy()
FORWARD = 5
BACK = 60
RANGE = range (-BACK,FORWARD)
dfRET['AbnRets'] = df[f'Rets']
for i in RANGE: dfRET[f'{i}'] = df[f'Rets'].shift(-i)
dfRET = dfRET[dfRET['SigmaLvl']>2]
plt.figure(figsize=(15,8))
plt.boxplot(dfRET.iloc[:,6:].values,showfliers=False)
plt.title('Returns')
plt.show()

#Check MAE's of absolute returns for positive outliers:
for DAYS in RANGES:
    dfAB = data0['2010-1-1':].copy()
    FORWARD = 5
    BACK = 60
    RANGE = range (-BACK,FORWARD)
    dfAB['AbnRets'] = df[f'{DAYS}D AbnRet']
    for i in RANGE: dfAB[f'{i}'] = df[f'{DAYS}D AbnRet'].shift(-i)
    dfAB = dfAB[abs(dfAB['SigmaLvl'])>2 ].dropna()
    print("Est. window %4i days, before outlier MAE: %6.4f, after outlier MAE: %6.4f"%( DAYS, np.average(abs(dfAB.iloc[:,5:-5]).values), np.average(abs(dfAB.iloc[:,-4:]).values) ))


# Check Relevance of Beta estimation Window
for DAYS in RANGES:
    dfAB = data0['2010-1-1':].copy()
    FORWARD = 5
    BACK = 60
    RANGE = range (-BACK,FORWARD)
    dfAB['AbnRets'] = df[f'{DAYS}D AbnRet']
    for i in RANGE: dfAB[f'{i}'] = df[f'{DAYS}D AbnRet'].shift(-i)
    dfAB = dfAB[dfAB['SigmaLvl']>2]
    plt.figure(figsize=(15,8))
    plt.boxplot(dfAB.iloc[:,5:].values,showfliers=False)
    plt.title(f'{DAYS} days beta calibration, Abnormal Returns')
    plt.show()

# Check Relevance of Beta estimation Window
DAYS = 180
dfAB = data0['2010-1-1':].copy()
FORWARD = 5
BACK = 60
RANGE = range (-BACK,FORWARD)
dfAB['AbnRets'] = df[f'{DAYS}D AbnRet']
for i in RANGE: dfAB[f'{i}'] = df[f'{DAYS}D AbnRet'].shift(-i)
dfAB = dfAB[dfAB['SigmaLvl']>2]
plt.figure(figsize=(15,8))
plt.boxplot(dfAB.iloc[:,5:].values,showfliers=False)
plt.title(f'{DAYS} days beta calibration, Abnormal Returns')
plt.show()

# Abnormal Returns, 180 days, smaller range
DAYS = 180
dfAB = data0['2010-1-1':].copy()
FORWARD = 5
BACK = 10
RANGE = range (-BACK,FORWARD)
dfAB['AbnRets'] = df[f'{DAYS}D AbnRet']
for i in RANGE: dfAB[f'{i}'] = df[f'{DAYS}D AbnRet'].shift(-i)
dfAB = dfAB[dfAB['SigmaLvl']>2]
plt.figure(figsize=(15,8))
plt.boxplot(dfAB.iloc[:,5:].values,showfliers=False)
plt.title(f'{DAYS} days beta calibration, Abnormal Returns')
plt.show()

# Check boxplot of cumulative returns   
RANGE = range (-BACK,FORWARD)
dfCAB = dfAB.copy()
for i in RANGE: dfCAB[f'{i}'] = dfCAB[f'{i}'] + 1
for i in range (-BACK,FORWARD-1): dfCAB[f'{i+1}'] = dfCAB[f'{i+1}'] * dfCAB[f'{i}']
plt.figure(figsize=(15,8))
plt.boxplot(dfCAB.iloc[:,5:].values,showfliers=False)
plt.title(f'{DAYS} days beta calibration, Cumulative Abnormal Returns')
plt.show()


## Plotting Betas
#fig, ax = plt.subplots()
#for counter in range( len(COLS) ) :
#    plt.plot(df[f'{COLS[counter]}Beta'].iloc[PLOTRANGE:], linewidth = 1, label = LABELS[counter], alpha = ALPHA[counter])
#plotoutliers(DATESTART)
#plt.title('Beta estimations with different ranges')
#fig.autofmt_xdate()
##plt.legend()
#plt.show()
#
## Plotting Alphas
#fig, ax = plt.subplots()
#for counter in range( len(COLS) ) :
#    plt.plot(df[f'{COLS[counter]}Alpha'].iloc[PLOTRANGE:], linewidth = 1, label = LABELS[counter], alpha = ALPHA[counter])
#plt.title('Alpha estimations with different ranges')
#fig.autofmt_xdate()
#plotoutliers(DATESTART)
##plt.legend()
#plt.show()
#
#
## Plotting Abnormal Returns
#fig, ax = plt.subplots()
#for counter in [1,3,9]: #range(0, len(COLS), 2 ) :
#    plt.plot(df[f'{COLS[counter]} AbnRet'].iloc[PLOTRANGE:], linewidth = 1, label = LABELS[counter], alpha = ALPHA[counter])
#plt.title('Abnormal Returns ($Y_i - \hat{Y}_i$) with different \u03B2 estimations')
#fig.autofmt_xdate()
#plotoutliers(DATESTART)
##plt.legend()
#plt.show()

# To test
#df1['Sum of last 5'] = pd.Series([df1['Rets'][i-5:i].sum() for i in range(len(df1))],index=df1.index)
# This returns 5 range of outliers
# df1['ReturnsUnderanOutlier'] = pd.Series( [ i if any(df1['SigmaLvl'].iloc[i : i + 5] > 0) else 0 for i in range(len(df))] , index = df1.index)
# This puts to another column but unable to use it since outliers are overlapping. can use to doublecheck.
#df1.loc[df1.loc[:dfP.index[0],'ReturnsUnderanOutlier'][-5:].index[0]:dfP.index[0],'ReturnsUnderanOutlier'] = dict( pd.Series(df1.loc[:dfP.index[0],'Rets'][-5:].cumsum(), name = 'ReturnsUnderanOutlier') )
#This returns with dates in the dict
#dfP['ABNRETS'] = [dict(df1.loc[:dfP.index[i],'Rets'][-5:].cumsum()) for i in range(len(dfP))] # , name = 'ReturnsUnderanOutlier'
# This gives with dates
#dfP['ABNRETS'] = [dict(df1.loc[:dfP.index[i],'Rets'][-5:].cumsum()) for i in range(len(dfP))] # , name = 'ReturnsUnderanOutlier'

#This is only for before slices.
#dfP['ABNRETS'] = [ df1.loc[:dfP.index[i],'Rets'] # This Returns a dataset until a specific date. 
#                    [-BACK:].cumsum().values for i in range(len(dfP))]  # [-BACK] will get the specific date interval.


## Calculating cumulative sum abnormal returns
#FORWARD = 5; BACK = 5 ; TOT = FORWARD + BACK
#df1 = data0.copy()
#MASKS = pd.DataFrame(dfP.index, index = range( len(dfP.index))) # This has to be a simple dataframe
#MASKS.columns = ['ActualDate']
#MASKS['ED']  = pd.Series([df1.loc[ MASKS['ActualDate'].loc[i]:] .iloc[BACK:BACK +1,:].index[0]          for i in range(1, len(MASKS)-1 ) ])
#MASKS['SD']  = pd.Series([df1.loc[:MASKS['ActualDate'].loc[i]]  .iloc[-FORWARD:-FORWARD +1,:].index[0]  for i in range(1, len(MASKS)-1 ) ])
##len(df1.loc[MASKS['StartDate'].loc[0] : MASKS['EndDate'].loc[0]])
#dfP['ABNRETS'] = pd.Series([ df1.loc[MASKS['SD'].loc[i] : MASKS['ED'].loc[i]]['Rets'].cumsum() for i in range(len(dfP.index)) ], index = dfP.index)
#len(dfP['ABNRETS'].values[0]) # Each ABNRET value has FORWARD + END length
#
#for i in range(0, 5): #len(dfP.index) -2 ):
#    plt.plot(range(0,TOT) , dfP['ABNRETS'][i].values, label = dfP['ABNRETS'][i].index[FORWARD-1])
#plt.legend();plt.show()
#for i in range(0,len(dfP.index)-2):
#    plt.scatter(range(0,TOT) , dfP['ABNRETS'][i].values )
#plt.show()
#dfP1 = dfP.copy()
#for STEPS in range(0,TOT):
#    dfP1[f'{STEPS - FORWARD}'] = pd.Series([ dfP1['ABNRETS'][OUTLIER].values[STEPS] for OUTLIER in range(0,len(dfP1.index)-2) ], index = dfP1.index[:-2])
#
#plt.boxplot(dfP1.iloc[:,4:].dropna().values)
#plt.show()
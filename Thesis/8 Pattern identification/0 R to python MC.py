#https://rpy2.readthedocs.io/en/version_2.8.x/introduction.html
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
data = web.DataReader("SWED-A.ST", "yahoo", start = "2000")['Adj Close']

def ArimaMC(data, simlen = 100, simtimes = 10):
    import rpy2.robjects.packages as rpackages
    utils = rpackages.importr('utils')
    utils.install_packages('quantmod')
    #qmod = rpackages.importr('quantmod')
    utils.install_packages('forecast')
    fcast = rpackages.importr('forecast')
    import rpy2.robjects as robjects
    autoarima = robjects.r['auto.arima']
    simulate = robjects.r['simulate']
    coef = robjects.r['coef']
    asnumeric = robjects.r['as.numeric']
    from rpy2.robjects import pandas2ri

    r_dataframe = pandas2ri.py2ri(data)
    modelR = autoarima(r_dataframe)
    #print(modelR);coef(modelR)
    montecarlo = [np.array(asnumeric(simulate(modelR,nsim = 100))) for i in range(10)]
    #plt.plot(np.transpose(montecarlo))
    return np.transpose(montecarlo)
plt.plot(data)
montecarlo = ArimaMC(data, simlen = 100, simtimes = 10)
import pandas as pd
plt.plot(pd.date_range(start="2019-05-29", periods=len(montecarlo)),montecarlo)
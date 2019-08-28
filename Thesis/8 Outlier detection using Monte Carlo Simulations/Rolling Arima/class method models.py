# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:33:24 2019

@author: Qognica
"""
import AlpsFuncs
from pmdarima.arima import auto_arima
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 
from datetime import timedelta 

class Examplemodel(object):
    def __init__(self,ticker,data=None, plot=True):
        self.ticker = ticker
        self.data = data
        if plot==True: plt.plot(self.train);plt.title(ticker);plt.show()
        
    def fitmodel(self, modeltype):
        self.modeltype = modeltype
        
        if modeltype == 'pmdARIMA':
            self.model = auto_arima(self.data,   #trace=True,
                       error_action='ignore',  # don't want to know if an order does not work
                       suppress_warnings=True,  # don't want convergence warnings
                       stepwise=True,trace=False)
            self.insamplepreds = pd.DataFrame(mymodel.model.predict_in_sample(),index=mymodel.data.index)
            self.RMSarray = pd.DataFrame(np.power(mymodel.model.resid(),2),columns=['RootMeanSquared'],index=mymodel.data.index)[5:]#RootMeanSquare indicate the goodness of fit
            self.RMS=float(np.sqrt(np.mean(self.RMSarray)))
            self.MPEarray = pd.DataFrame((mymodel.model.resid()/mymodel.data.values),columns=['MPE'],index=mymodel.data.index)[5:]#MeanPerctError indicate the goodness of fit
            self.MPE=float(np.mean(self.MPEarray)*100)
            self.MAPEarray = pd.DataFrame((abs(mymodel.model.resid())/mymodel.data.values),columns=['MAPE'],index=mymodel.data.index)[5:]#MeanPerctError indicate the goodness of fit
            self.MAPE=float(np.mean(self.MAPEarray)*100)
            #rmse = sqrt(mean_squared_error(df['temperature'][-7:], forecast))
    def pred(self, predictionlen, OutofSampleData=None):
            #Predictions
            #To predict custom dates:
            #predictionindex = pd.date_range(start = mymodel.data.index[-1:][0], end= (mymodel.data.index[-1:][0] + timedelta(days=predictionlen-1)) )
            predictionindex = OutofSampleData.index
            self.Predictions = pd.DataFrame(mymodel.model.predict(n_periods=predictionlen, return_conf_int=True)[0], columns=['Predictions'],index = predictionindex)
            self.confint = pd.DataFrame(mymodel.model.predict(n_periods=predictionlen, return_conf_int=True)[1], columns=['Lower','Upper'], index = predictionindex)

            if OutofSampleData is not None:
                self.OutofSampleData=pd.DataFrame(OutofSampleData, index = predictionindex)
                if predictionlen < len(self.OutofSampleData):
                    self.OutofSampleData = self.OutofSampleData[:predictionlen]
                if predictionlen > len(self.OutofSampleData):
                    self.Predictions = self.Predictions[:len(self.OutofSampleData)]
                self.OutofSampleRMS = np.power(self.Predictions - self.OutofSampleData.values,2)
    #def cleanparameters(self):
    #    self.allparams = pd.DataFrame(columns=['RootMeanSquare','MeanPercError','MeanAbsError','ARorders','I orders','MAorders','AR1','AR2','AR3','AR4','AR5','MA1','MA2','MA3','MA4','MA5'])
        
    def addparameters(self,i):
        #Orders
        ARorder=mymodel.model.order[0];Iorder=mymodel.model.order[1];MAorder=mymodel.model.order[2]
        currentparams = pd.DataFrame([[self.RMS,self.MPE,self.MAPE,ARorder,Iorder,MAorder]],
                                     columns=['RootMeanSquare','MeanPercError','MeanAbsError','ARorders','I orders','MAorders'],index = [i])
        # ARx and MAx coefficients
        for arcounter in range(1,6):
            try:currentparams['AR%i'%(arcounter)] = mymodel.model.arparams()[arcounter-1]
            except:currentparams['AR%i'%(arcounter)] = 0
        for macounter in range(1,6):
            try: currentparams['MA%i'%(macounter)] = mymodel.model.maparams()[macounter-1]
            except: currentparams['MA%i'%(macounter)] = 0
        self.currentparams = currentparams
        
        #if not 'mymodel.allparams' in locals():
        #    self.cleanparameters()
        #self.allparams = self.allparams.append(currentparams,sort=False)
        return
def plot(mymodel,savefile=False,i = 1,iterations = 1,slicestep = 1,steps = 1):
    #AlpsFuncs.plotwithslices(data,figsize =(14,6),slices=10,rollingwidth=1,Testoutput = 0,stdwidth = 0.5,doubleaxis=False)
    fig, ax1=plt.subplots()
    plt.plot(mymodel.data,              color='k',linewidth=1,label='Swedbank Price')
    #plt.plot(mymodel.insamplepreds,     color='c',linestyle=':',linewidth=1,label='In Sample Predictions') # insample prediction
    plt.plot(mymodel.Predictions,       color='b',linestyle='-',linewidth=1,label='Predictions');plt.grid() #insample errors
    plt.plot(mymodel.OutofSampleData,   color='Silver',linestyle='-',linewidth=2) #insample errors
    plt.fill_between(mymodel.confint.index,mymodel.confint['Lower'],mymodel.confint['Upper'], color='k', alpha=0.5)
    plt.axvline(mymodel.Predictions.index[0:1],color='b',linestyle='--',linewidth=1, label = 'Training Limit')
    plt.legend(loc = 2);plt.grid();ax2 = ax1.twinx()
    plt.plot(mymodel.OutofSampleRMS,    color='r',linestyle=':',linewidth=1);plt.grid()
    #plt.plot(mymodel.MAPEarray,         color='r',linestyle=':',linewidth=1,label='MAPE Residuals. W/MAPE: %%%1.1f' %(mymodel.MAPE))
    plt.plot(mymodel.RMSarray,          color='r',linestyle=':',linewidth=1) #,label='In sample residuals'
    plt.legend(loc =6);
    try : paramone = mymodel.model.arparams()
    except : paramone = 0
    try : paramtwo = mymodel.model.maparams()
    except : paramtwo = 0
    #plt.title('T = %s   Iteration #  %i of %i, Split: %5.3f, Slicestep: %5.4f, Steps: %3.3f, \nParams: %s, Coefar: %s, CoefMa %s' 
    #          %(mymodel.ticker, i, iterations, (split + steps * i),slicestep,steps,mymodel.model.order,paramone,paramtwo))
    plt.title('T = %s   Iteration #  %i of %i, Split: %5.3f, Stepsize: %3.3f, \nParams: %s, Coefar: %s, CoefMa %s' 
              %(mymodel.ticker, i, iterations, (split + steps * i),steps,mymodel.model.order,paramone,paramtwo))
    if savefile==True:
        plt.savefig('Iteration  = %3.0f, Params =%s.jpeg'%(i,mymodel.model.order))
    plt.show()
    #resids
    return
#burlywood
ticker = "SWED-A.ST"
totaldata  = web.DataReader(ticker, 'yahoo', start = "2010")['Adj Close']

AlpsFuncs.DetailsofData(totaldata)

split = 0.5
iterations = 500
steps = (1 - split )/ iterations
savefile = True

import os as os
os.chdir(r"C:\Users\Qognica\Dropbox\Qognica Master Thesis 2019\Work folder Alp\alp\Written Thesis reports\8 Rolling ARIMA identification\Savehere")

for i in range(1,int(iterations)):
    slicestep = int( len(totaldata) * (split + steps * i) )
    
    #Splitting training and testing sets
    train = totaldata[:slicestep]
    test = totaldata[slicestep:]
    
    #Fitting model
    mymodel = Examplemodel(ticker,data=train,plot=False)
    mymodel.fitmodel('pmdARIMA')
    
    #getting predictions and plotting
    mymodel.pred(OutofSampleData = test,predictionlen=len(test))
    plot(mymodel,savefile,i = i,iterations = iterations,slicestep = slicestep,steps = steps)

    #adding parameters
    mymodel.addparameters(i)
    if i == 1:  ParameterHistory = mymodel.currentparams
    else:       ParameterHistory = ParameterHistory.append(mymodel.currentparams,sort=False)

plt.savefig('Last Iteration')
#Saving Parameters
ParameterHistory.to_csv(r'ParameterHistory.txt')

#returns column name ParameterHistory.iloc[:,0]
#returns i orders ParameterHistory.iloc[:,2]
#number of columns len(list(ParameterHistory.columns))

#plt.plot(ParameterHistory['ARorders'],label = 'ARorders');plt.legend();plt.title('ARorders');plt.savefig('ARorders');plt.show
#plt.plot(ParameterHistory['MAorders'],label = 'MAorders');plt.legend();plt.title('MAorders');plt.savefig('MAorders');plt.show
#plt.plot(ParameterHistory['AR1'],label = 'AR1 parameter');plt.legend();plt.title('AR1');plt.savefig('AR1 parameter');plt.show
#plt.plot(ParameterHistory['MA1'],label = 'MA1 parameter');plt.legend();plt.title('MA1');plt.savefig('MA1 parameter');plt.show
#plt.plot(ParameterHistory['MA2'],label = 'MA2 parameter');plt.legend();plt.title('MA2');plt.savefig('MA2 parameter');plt.show

fig = plt.figure(figsize=(15,12))
fig.subplots_adjust(hspace=0.7, wspace=0.2)
for i in range(1, len(list(ParameterHistory.columns))):
    # Coordinates of plots
    ax = fig.add_subplot(int(len(list(ParameterHistory.columns))/3)+1, 3, i)
    # Plot itself
    ax.plot(ParameterHistory.iloc[:,i])
    ax.set_xlabel(list(ParameterHistory.columns)[i])
plt.savefig('Params')

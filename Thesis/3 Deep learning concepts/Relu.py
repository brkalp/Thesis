import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pandas_datareader as web
import seaborn as sns
sns.set()

random.seed(42)
x = np.arange(-10,10)
y = np.max([np.zeros(len(x)), x], axis=0)
z = np.array([ i if i>0 else i*0.01 for i in x ])
v = np.array([ i if i>0 else i*0.5 for i in x ])

plt.plot( x, x ,  label = ('Input') , color = 'k', linestyle =':', linewidth =2)
plt.plot( x, y ,  label = ('ReLU') , color = 'b', linestyle =':', linewidth =2); plt.legend();plt.show()

plt.plot( x, x , label = ('Input') , color = 'k', linestyle =':', linewidth =2)
plt.plot( x, z , label = ('Leaky ReLU') , color = 'b', linestyle =':', linewidth =2); plt.legend();plt.show()

plt.plot( x, x , label = ('Input') , color = 'k', linestyle =':', linewidth =2)
plt.plot( x, v , label = ('parametric  ReLU with $a$ = 0.5') , color = 'b', linestyle =':', linewidth =2); plt.legend();plt.show()


# coding: utf-8

# In[1]:


import scipy as sp
import numpy as np
import statsmodels.formula.api as sm
import pandas as pd


# In[2]:


# Валентина Садова
# множественная регрессия


# In[10]:


x1 = np.random.normal(0,1,20)
x2 = np.random.normal(0,3,20)
y = 3 + 2*x1 - x2 + np.random.normal(0,0.5,20)


# In[11]:


z = sm.ols(formula ='y ~ x1 + x2', data=pd.DataFrame({'y':y,'x1':x1,'x2':x2}))
z_fit = z.fit()
z1 = z_fit.summary()


# In[12]:


z1


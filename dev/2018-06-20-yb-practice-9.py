
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


import numpy as np
import pandas as pd
import scipy.stats as sts
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style('whitegrid')


# In[3]:


x1 = np.random.normal(0,1,20)
x2 = np.random.normal(0,3,20)
y = 3 + 2*x1 - x2 + np.random.normal(0,0.5,20)


# In[4]:


z = sm.ols(formula ='y ~ x1 + x2', data=pd.DataFrame({'y':y,'x1':x1,'x2':x2}))
z_fit = z.fit()


# In[23]:


z_fit.summary()


# In[38]:


pd.Series([3, 1], index=['x1', 'x2'])


# In[49]:


# z_fit.predict(pd.Series([3, 1], index=['x1', 'x2']))
z_fit.predict({'x1': [10], 'x2': [3]})


# In[45]:


xast = np.array([10, 3, 1])
B = pd.DataFrame({'y':y,'x1':x1,'x2':x2})[['x1', 'x2']]
B['intercept'] = 1
B.values


# In[48]:


d = 1 + xast.dot(np.linalg.inv(B.T.dot(B))).dot(xast.T)
d



# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')

sns.set(rc={'figure.figsize':(8,6)})
sns.set_style('whitegrid')


# In[19]:


normal = norm(0, 1)
source = norm(1, 2)


# In[17]:


source.var()


# In[18]:


source.std()


# In[20]:


sample = source.rvs(10)
sample


# In[6]:


_ = sns.distplot(sample)


# In[21]:


lower_clt = sample.mean() - normal.ppf(1-(1-0.95)/2) * source.var() / np.sqrt(sample.size)
lower_clt


# In[22]:


upper_clt = sample.mean() + normal.ppf(1-(1-0.95)/2) * source.var() / np.sqrt(sample.size)
upper_clt


# In[9]:


large_sample = source.rvs((10000, 10))
means = large_sample.mean(axis=1)
means


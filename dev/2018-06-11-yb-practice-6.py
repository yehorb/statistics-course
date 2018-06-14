
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')

sns.set(rc={'figure.figsize':(8,6)})
sns.set_style('whitegrid')


# In[40]:


x = sts.norm.rvs(0, 4, (1000, 10))
x


# $$
# \begin{align}
# H_0: \mu & = 3 \\
# H_1: \mu & \neq 3
# \end{align}
# $$

# In[4]:


a = 0.05
z = sts.norm.ppf(1 - a/2)
z


# $$
# T(\chi) = \frac{\sqrt{n} \; |\hat \mu - \mu|}{\sigma}
# $$

# In[5]:


t_test = lambda mu_, sigma, x: np.sqrt(x.size) * np.abs(x.mean() - mu_) / sigma


# In[6]:


trues = np.fromiter(map(lambda arr: t_test(0, 4, arr) > z, x), dtype=np.bool)
trues.sum()


# $$
# T(\chi) = \frac{\sqrt{n} \; |\hat \mu - \mu|}{\sigma}
# $$
# With $\sigma$ unknown, we calculate it from observed distribution and use T dist for tests.

# In[44]:


x_ = sts.chi2.rvs(10, size=(1000, 100))
x_


# In[24]:


z_test = lambda x:  sts.t.ppf(1 - a/2, x.size - 1)
z_test(x_[0])


# In[25]:


t_test_ = lambda mu_, x: np.sqrt(x.size) * np.abs(x.mean() - mu_) / x.std()


# In[26]:


arr = x_[75]
txt = 'h0' if t_test_(10, arr) > z_test(arr) else 'h1'
txt


# In[31]:


trues_ = np.fromiter(map(lambda arr: t_test_(10, arr) < z_test(arr), x_), dtype=np.bool)
trues_.sum()


# In[34]:


pdf = lambda x: sts.chi2.pdf(x, 10)


# In[51]:


a = sts.kstest(x[0], 'norm')
a


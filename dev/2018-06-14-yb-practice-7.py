
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[3]:


n = 100
actual = np.array([22, 53, 25])
# prob_m = (actual * [2, 1, 0] / (n * 2)).sum()
prob_m = 0.5
prob_i = np.array([prob_m * prob_m, 2 * prob_m * (1 - prob_m), (1 - prob_m) * (1 - prob_m)])
expect = prob_i * n
# chi2_c = ((actual - expect) ** 2 / expect).sum()
sts.chisquare(actual, expect)


# In[4]:


n = 100
actual = np.array([22, 33, 45])
prob_m = (actual * [2, 1, 0] / (n * 2)).sum()
prob_i = np.array([prob_m * prob_m, 2 * prob_m * (1 - prob_m), (1 - prob_m) * (1 - prob_m)])
expect = prob_i * n
# chi2_c = ((actual - expect) ** 2 / expect).sum()
# в сложной теореме уменьшаем количество степеней свободы. в даном случае остается только prob_m как независимая величина
# потому оставляем ddof = 1 (dof = k - 1 - ddof)
sts.chisquare(actual, expect, ddof=1)


# In[5]:


# no expect -- tests for discrete uniform dist
sts.chisquare(actual, ddof=1)


# In[6]:


1 - sts.chi2.cdf(9.189136689774069, 2)


# In[7]:


observed = sts.norm.rvs(0, 1, 100)
observed


# In[8]:


bins = np.histogram(observed, bins='auto')
sns.distplot(observed, bins=bins[1], kde=False)


# In[9]:


bins


# In[10]:


prob_bins = list(bins[1])
prob_bins[0] = -np.inf
prob_bins[-1] = np.inf
prob_bins


# In[11]:


# для сложных гипотез заменить (0, 1) параметрами получеными из выборки (не _.std())
func = sts.norm(0, 1).cdf
prob_1 = func(prob_bins[:-1])
prob_2 = func(prob_bins[1:])
prob = prob_2 - prob_1
prob


# In[12]:


expected = prob * observed.size
expected


# In[13]:


# соответственно здесь уменьшаем количество степеней свободы на 2 (ddof=2)
assert bins[0].size == expected.size
sts.chisquare(bins[0], expected)


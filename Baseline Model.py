#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np

import sklearn
from sklearn import metrics


# In[15]:


data = pd.read_csv('D:\Python Projects\Thesis\Data\Final.csv', sep=',')

data  = data[['month', 'cloud_coverage', 'rainfall', 'range_temp', 'wind_speed', 'relative_humidity', 'avg_wl(m)', 'nxt_avg_wl(m)']]
predict = "nxt_avg_wl(m)"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,)


# In[18]:


y_test


# In[19]:


y_train


# In[29]:


predict = y_train.mean()
predict


# In[30]:


predict2 = [predict] * len(y_train)


# In[31]:


print ('Mean Squared Error: ', metrics.mean_squared_error(y_train, predict2))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_train, predict2)))
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_train, predict2))
print('R Squared: ', metrics.r2_score(y_train, predict2))


# In[ ]:





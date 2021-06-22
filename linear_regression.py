#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np

import sklearn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import style


# In[61]:


data = pd.read_csv('D:\Python Projects\Thesis\Data\Final.csv', sep=',')

data  = data[['month', 'cloud_coverage', 'rainfall', 'range_temp', 'wind_speed', 'relative_humidity', 'avg_wl(m)', 'nxt_avg_wl(m)']]
predict = "nxt_avg_wl(m)"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[62]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,)
linear = LinearRegression()
linear.fit(x_train, y_train)

test_acc = linear.score(x_test, y_test)
train_acc = linear.score(x_train, y_train)

cvs = cross_val_score(LinearRegression(), x, y, cv = 5)
print('Test Accuracy: ', test_acc*100, '%')
print('Train Accuracy: ', train_acc*100, '%')
print('Cross validation score: ', cvs)
print('Coeffecient: ', linear.coef_)
print('Intercept: ', linear.intercept_)
print('\n')

predictions = linear.predict(x_test)
print ('Mean Squared Error: ', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, predictions))
print('R Squared: ', metrics.r2_score(y_test, predictions))
print('\n')


for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# In[ ]:





# In[ ]:





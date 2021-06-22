#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style


data = pd.read_csv('D:\Python Projects\Thesis\Data\Final.csv', sep=',')

data.head()


# In[75]:


data = pd.read_csv('D:\Python Projects\Thesis\Data\Final.csv', sep=',')

data  = data[['range_temp', 'wind_speed', 'cloud_coverage', 'avg_wl(m)', 'max_wl(m)', 'probability', 'nxt_probability']]
predict = "nxt_probability"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[76]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
model = GaussianNB()
model.fit(x_train,y_train)


# In[77]:


test_acc = model.score(x_test, y_test)
train_acc = model.score(x_train, y_train)

print('Test Accuracy: ', test_acc*100, '%')
print('Train Accuracy: ', train_acc*100, '%')
print('\n')

predictions = model.predict(x_test)

print ('Mean Squared Error: ', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, predictions))
print('R Squared: ', metrics.r2_score(y_test, predictions))
print('\n')

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# In[ ]:





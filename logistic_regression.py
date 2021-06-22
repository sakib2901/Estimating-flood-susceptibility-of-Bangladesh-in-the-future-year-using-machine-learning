#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import style


# In[95]:


data = pd.read_csv('D:\Python Projects\Thesis\Data\Final.csv', sep=',')

data  = data[['month', 'max_temp', 'rainfall', 'relative_humidity', 'cloud_coverage', 'wind_speed', 'probability']]
predict = "probability"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[96]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 0)
logreg = LogisticRegression(solver='lbfgs', max_iter=500)
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
for x in range(len(y_pred)):
    print(y_pred[x], x_test[x], y_test[x])


# In[97]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:





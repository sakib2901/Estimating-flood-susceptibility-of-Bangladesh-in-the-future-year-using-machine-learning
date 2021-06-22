#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('D:\Python Projects\Thesis\Data\Final.csv', sep=',')

data.head()


# In[2]:


data  = data[['range_temp', 'wind_speed', 'cloud_coverage', 'avg_wl(m)', 'max_wl(m)', 'probability', 'nxt_probability']]
predict = "nxt_probability"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[4]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train.shape


# In[5]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# In[6]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, bias_initializer = 'he_uniform',activation='relu',input_dim = 6))

# Adding the second hidden layer
classifier.add(Dense(units = 6, bias_initializer = 'he_uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(units = 1, bias_initializer = 'glorot_uniform', activation = 'sigmoid'))


# In[7]:


classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[8]:


model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 100)


# In[9]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[10]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[11]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)


# In[12]:


print(score)


# In[13]:


from sklearn import metrics
print ('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('R Squared: ', metrics.r2_score(y_test, y_pred))


# In[15]:


for x in range(len(y_pred)):
    print(y_pred[x], X_test[x], y_test[x])


# In[ ]:





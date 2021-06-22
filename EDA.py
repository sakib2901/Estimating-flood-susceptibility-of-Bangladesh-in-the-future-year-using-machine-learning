#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
df1 = pd.read_csv('D:\Python Projects\Thesis\Data\water.csv')
df2 = pd.read_csv('D:\Python Projects\Thesis\Data\weather.csv')

df1.head()


# In[2]:


df2.head()


# In[32]:


df3 = pd.merge(df1, df2, how='inner')

df3.head()


# In[33]:


df3.shape


# In[34]:


df3.to_csv("D:\Python Projects\Thesis\Data\Merged.csv", columns = ['Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Monthly_Max(m)', 'Monthly_Min(m)', 'Monthly_Avg(m)', 'Period'], index = False)


# In[4]:


data = pd.read_csv('D:\Python Projects\Thesis\Data\Final.csv')
data = data[['month', 'cloud_coverage', 'rainfall', 'wind_speed', 'relative_humidity','max_temp', 'range_temp', 'max_wl(m)', 'nxt_max_wl(m)', 'avg_wl(m)', 'nxt_avg_wl(m)']]


# In[5]:


data.head()


# In[6]:


data.isnull().sum()


# In[7]:


corelation = data.corr()


# In[8]:


sns.heatmap(corelation,xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)


# In[15]:


sns.pairplot(data)


# In[3]:


sns.relplot(x='nxt_max_wl(m)', y='relative_humidity', data=data)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np

data=pd.read_csv('house_price_data.txt', sep=',' , header=None, names=['X1', 'X2', 'Y'])
data.head(10)


# In[2]:


X_avg = int(np.sum(data.X1) / data.X1.size)
Y_avg = int(np.sum(data.Y) / data.Y.size)

X_dino = np.max(data.X1) - np.min(data.X1)
Y_dino = np.max(data.Y) - np.min(data.Y)

data = data.assign(X3 = lambda x: (x['X1'] - X_avg)/X_dino)
data = data.assign(Y2 = lambda x: (x['Y'] - Y_avg)/Y_dino)
data_normalized = data.drop(columns = ['X1','X2','Y'])

data_normalized.head(10)


# In[3]:


sns.scatterplot(data=data, x="X3", y="Y2")


# In[4]:


# split a dataset into train and test sets

from sklearn.model_selection import train_test_split

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(data.X3, data.Y2, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = np.array(X_train)
y_train = np.array(y_train)


# In[5]:


theta1, theta2 = 0.3,0.2
cost_history =[]

for i in range(500):
    j=0
    dj_by_dtheta1 =0
    dj_by_dtheta2 =0
    m = len(X_train)
    
    for k in range(m):
        j = j + ((theta1 * X_train[k] + theta2 - y_train[k])**2) / (2*m)
        dj_by_dtheta1 = dj_by_dtheta1 + (theta1 * X_train[k] + theta2 - y_train[k]) * X_train[k] / m
        dj_by_dtheta2 = dj_by_dtheta2 + (theta1 * X_train[k] + theta2 - y_train[k]) / m
        
    print('cost in ',i,'th iteration is ',j,' and theta values are',theta1,theta2)
    
    theta1 = theta1 - dj_by_dtheta1 * 0.1
    theta2 = theta2 - dj_by_dtheta2 * 0.1
    
    cost_history.append(j)
        


# In[6]:


X_plot = np.linspace(-0.4, 0.8, len(X_train))
Y_plot = theta1*X_plot+ theta2


# In[7]:


df1 = pd.DataFrame(columns=['X_train','y_train'])
df2 = pd.DataFrame(columns = ['X_plot','Y_plot'])
df1.X_train = X_train
df1.y_train = y_train
df2.X_plot = X_plot
df2.Y_plot = Y_plot


# In[8]:


sns.scatterplot(data=df1, x="X_train", y="y_train")
sns.lineplot(data=df2, x='X_plot', y='Y_plot', color='g')


# In[9]:


cost_hist = pd.DataFrame(columns =['cost_history'])
cost_hist.cost_history = cost_history

sns.lineplot(data=cost_hist, x=cost_hist.index, y='cost_history', color='g')


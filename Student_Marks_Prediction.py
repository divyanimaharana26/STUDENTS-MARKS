#!/usr/bin/env python
# coding: utf-8

# # Predict the percentage of an student based on the no. of study hours.
# 

# # Author- Divyani Maharana

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("student.csv",sep=(","))
df


# In[6]:


df.isnull().sum()


# In[14]:


df.drop(df.tail(3).index,inplace=True)


# In[43]:


df


# In[44]:


X=df.drop(["Scores"],axis=1)
y=df['Scores']


# In[45]:


y.shape


# In[46]:


X.shape


# In[47]:


from sklearn.model_selection import train_test_split


# In[163]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)


# In[164]:


X_train.shape


# In[165]:


X_test.shape


# In[166]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[167]:


lr.intercept_


# In[168]:


lr.coef_


# In[169]:


pred=lr.predict(X_test)
pred


# In[170]:


y_test


# In[171]:


lr.score(X_test,y_test)*100


# In[172]:


df_new=pd.DataFrame(np.c_[X_test,y_test,pred],columns=['Study hrs','Marks Original','Marks Predicted'])


# In[173]:


df_new


# In[174]:


plt.scatter(X_train,y_train)


# In[175]:


plt.scatter(X_test,y_test)
plt.plot(X_train,lr.predict(X_train),color='r')


# In[ ]:





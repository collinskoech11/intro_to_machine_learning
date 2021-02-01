#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import dependencies
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd


# In[2]:


# load dataset
data_set = pd.read_csv('Salary_Data.csv')


# we need to extract the dependent and independent variables from the given dataset. The independent variable is years of experience, and the dependent variable is salary. Below is code for it:!

# In[3]:


x = data_set.iloc[:, :-1].values # years of experience?
y = data_set.iloc[:, 1].values # salary?
print(x[:5])
print(y[:5])


# ## Splitting the dataset into training and test set

# In[4]:


# Splitting the dataset into training and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)  


# ## Fitting the Simple Linear Regression model to the training dataset  

# In[5]:


# Fitting the Simple Linear Regression model to the training dataset

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  
regressor.fit(x_train, y_train)  


# ## Prediction of Test and Training set result 

# In[6]:


# Prediction of Test and Training set result  
y_pred = regressor.predict(x_test)  
x_pred = regressor.predict(x_train)

print(x_pred, y_pred)


# ## visualizing the Training set results

# In[7]:


mtp.scatter(x_train, y_train, color="green")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Salary vs Experience (Training Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()


# ## visualizing the Test set results

# In[8]:


# visualizing the Test set results  
mtp.scatter(x_test, y_test, color="blue")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Salary vs Experience (Test Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()  


# In[ ]:





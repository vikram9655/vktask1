#!/usr/bin/env python
# coding: utf-8

# Data Science Project @Sparks Foundation (GRIP April2022)
# Task-1 Score prediction using Supervised ML
# Author - Vikram Kumar

# In[ ]:


# Importing necessary Libraries

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("data1.csv")


# In[6]:


data.head()


# In[8]:


data.shape


# In[9]:


data.count()


# In[10]:


data.describe()


# In[11]:


x= np.array(data['Hours']).reshape(-1, 1)
y = np.array(data['Scores']).reshape(-1,1)


# In[12]:


# Plotting Distribution of Hours and Scores

plt.figure(figsize=(10,6))
plt.title("No Hours studied VS Percentage Score", fontsize="large")
plt.scatter(x, y, color='green', label= 'Score')
plt.xlabel("No of Hours Studied", fontsize= "large")
plt.ylabel("Percentage Score", fontsize= "large")
plt.legend()
plt.show()


# In[13]:


data.corr()


# In[14]:


data.isnull()


# In[20]:


sns.heatmap(data.corr(),)


# In[21]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print("Data is successfully Splitted")


# In[22]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

print("Training Completed")


# In[23]:


print('Intercept value is:',model.intercept_)
print('Linear coefficient is:',model.coef_)


# In[24]:


#Fittig data into regression line

line = model.coef_*x + model.intercept_
plt.scatter(x,y, color="green", label="Actual Values")
plt.plot(x, line, label="Trained values")
plt.xlabel("No of Hours Studied", fontsize= "large")
plt.ylabel("Percentage Score", fontsize= "large")
plt.legend()
plt.show()


# In[25]:


# Predicting Model Score based on Test Data Set

y_pred = model.predict(x_test)

# plotting actual v/s Predicted values
plt.plot(x_test, y_pred, color="green", label='Predicted Values')

plt.scatter(x_test, y_test, label='Actual Values')
plt.xlabel("No of Hours Studied", fontsize= "large")
plt.ylabel("Percentage Score", fontsize= "large")

plt.legend()

plt.show()


# In[26]:


hours= [[9.25]]
pred_score = model.predict(hours)
print(f"Number of hours = {hours}")  
print(f"Prediction Score = {pred_score}")


# In[27]:


# Plotting predicted value on the graph

plt.plot(x_test, y_pred, color="green", label='Predicted Values')
plt.scatter(hours, pred_score, marker ='o', c='r', label='Predicted Score')
plt.scatter(x_test, y_test, label='Actual Values')
plt.xlabel("No of Hours Studied", fontsize= "large")
plt.ylabel("Percentage Score", fontsize= "large")
plt.legend()
plt.show()


# In[28]:


# Calculation of model score
model.score(x_test,y_test)


# Interpretation :- 94.35% variation in dependent variable(Scores) are explained by independent variable(Hours) in the model.
# 
# Conclusion
# For a student studying 9.25Hrs a day , the model predicts his score as 92.67022

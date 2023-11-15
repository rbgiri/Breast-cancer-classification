#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection & Processing

# In[2]:


# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


# In[3]:


print(breast_cancer_dataset)


# In[4]:


# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)


# In[5]:


# print the first 5 rows of the dataframe
data_frame.head()


# In[6]:


# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target


# In[7]:


# print last 5 rows of the dataframe
data_frame.tail()


# In[8]:


# number of rows and columns in the dataset
data_frame.shape


# In[9]:


# getting some information about the data
data_frame.info()


# In[10]:


# checking for missing values
data_frame.isnull().sum()


# In[11]:


# statistical measures about the data
data_frame.describe()


# In[12]:


# checking the distribution of Target Varibale
data_frame['label'].value_counts()


# 1 --> Benign
# 
# 0 --> Malignant

# In[13]:


data_frame.groupby('label').mean()


# Separating the features and target

# In[14]:


X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']


# In[15]:


print(X)


# In[16]:


print(Y)


# Splitting the data into training data & Testing data

# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[18]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# Logistic Regression

# In[19]:


model = LogisticRegression()


# In[20]:


# training the Logistic Regression model using Training data

model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[21]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


# In[22]:


print('Accuracy on training data = ', training_data_accuracy)


# In[23]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)


# In[24]:


print('Accuracy on test data = ', test_data_accuracy)


# Building a Predictive System

# In[25]:


input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')


# In[26]:


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize a Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)


# In[ ]:





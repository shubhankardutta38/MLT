#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split


# In[7]:



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
data = pd.read_csv(url, sep=';')
data


# In[8]:




# Load the wine quality dataset
#data = pd.read_csv('winequality-white.csv', delimiter=';')

# Split the dataset into training and testing sets
X = data[['alcohol']].values
y = data['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[9]:



# Create a linear regression object and fit the model to the training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the quality of the wine for the test data
y_pred = regressor.predict(X_test)

# Print the model's parameters
print('Coefficients: ', regressor.coef_)
print('Intercept: ', regressor.intercept_)

# Print the performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error: ', mse)
print('Root Mean Squared Error: ', rmse)
print('R-squared: ', r2)


# In[10]:




# Plot the learning curves
train_sizes, train_scores, test_scores = learning_curve(regressor, X, y, cv=5)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.title('Learning curves')
plt.legend()
plt.show()


# In[ ]:





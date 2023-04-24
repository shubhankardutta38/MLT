#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
data = pd.read_csv(url, sep=';')
data


# In[3]:


# Load the winequality-white dataset
# This example assumes that the dataset has already been preprocessed
# and is available as a pandas dataframe called "data"
# x_input is the input feature that we want to use for the sigmoid plot
x_input = data['fixed acidity']

# Define the logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))

# Calculate the y-values for the sigmoid plot
y_output = logistic(x_input)

# Create the plot
plt.plot(x_input, y_output)

# Set the plot title and labels
plt.title("Logistic Plot for Fixed Acidity")
plt.xlabel("Fixed Acidity")
plt.ylabel("Output")

# Show the plot
plt.show()


# In[ ]:





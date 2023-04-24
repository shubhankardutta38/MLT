#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[23]:


import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
data = pd.read_csv(url, sep=';')
data


# In[24]:


# Count the number of missing values in each column
print(data.isnull().sum())

# Drop any rows that contain missing values
data_dropped = data.dropna()

# Verify that there are no missing values in the new dataset
print(data_dropped.isnull().sum())


# In[25]:


# Impute missing values with the mean value of the column
data_mean = data.fillna(data.mean())

# Verify that there are no missing values in the new dataset
print(data_mean.isnull().sum())


# In[26]:



# Impute missing values with the median value of the column
data_median = data.fillna(data.median())

# Verify that there are no missing values in the new dataset
print(data_median.isnull().sum())


# In[ ]:


#scikit-learn's 
#SimpleImputer class, which provides different strategies for handling missing values.
#We will demonstrate how to use three different strategies:
#mean imputation, median imputation, and most frequent imputation.


# In[27]:


from sklearn.impute import SimpleImputer

# Create a SimpleImputer object with the mean strategy
mean_imputer = SimpleImputer(strategy='mean')

# Create a SimpleImputer object with the median strategy
median_imputer = SimpleImputer(strategy='median')

# Create a SimpleImputer object with the most frequent strategy
mode_imputer = SimpleImputer(strategy='most_frequent')

# Separate the target variable (quality) from the predictors (features)
X = data.drop('quality', axis=1)
y = data['quality']

# Fit the imputers to the data and transform the data
X_mean_imputed = mean_imputer.fit_transform(X)
X_median_imputed = median_imputer.fit_transform(X)
X_mode_imputed = mode_imputer.fit_transform(X)

# Verify that there are no missing values in the new datasets
print(pd.DataFrame(X_mean_imputed).isnull().sum())
print(pd.DataFrame(X_median_imputed).isnull().sum())
print(pd.DataFrame(X_mode_imputed).isnull().sum())


# In[28]:


import matplotlib.pyplot as plt

# Create a scatter plot of alcohol content vs. pH
plt.scatter(data['alcohol'], data['pH'])
plt.xlabel('Alcohol')
plt.ylabel('pH')
plt.show()


# In[29]:


import seaborn as sns

# Create a pair plot of all columns in the dataset
sns.pairplot(data)
plt.show()


# In[30]:


# Create a count plot of the "quality" column
sns.countplot(x='quality', data=data)
plt.show()


# In[31]:


import seaborn as sns

# Create a heatmap of the correlation between all columns in the dataset
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[44]:


from wordcloud import WordCloud

# Get column names of the dataset
column_names = data.columns.tolist()

# Create a word cloud plot of the column names
wordcloud = WordCloud(background_color='white').generate(' '.join(column_names))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Import Libraries and packages

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(
    { "figure.figsize": (8, 5) },
    style='ticks',
    color_codes=True,
    font_scale=1
)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# ML utilities
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn import metrics

# Models
from sklearn.cluster import KMeans

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# # Loading the dataset

# In[5]:


bank_df = pd.read_csv("../data/bank_data_C.csv")


# # Initial inspection of the dataset

# In[7]:


bank_df.head()


# In[8]:


bank_df.sample()


# In[9]:


bank_df.shape


# In[10]:


bank_df.info()


# ### The dataset contains 1041614 rows and 9 columns in which most of them are strings. The data types of the transaction date and customer date of birth were object rather than datetime.

# In[12]:


bank_df.describe(include='all')


# ### Based on the descriptive statistics, some abberations were observed in the dataset. For instance, customer gender had three unique values while the maximum date of birth was 1/1/1800.   

# In[14]:


# Checking Unique values in the columns
print("Unique Transaction:", bank_df['TransactionID'].nunique())
print("Unique Customers:",  bank_df['CustomerID'].nunique())
print("Customer Location:",  bank_df['CustLocation'].nunique())


# In[15]:


bank_df.isnull().sum()


# ### There are no null values (missing data)

# # Splitting the features to qualitative and quantitative features

# In[18]:


# Categorising quantitative features (numerical columns)
numerical_features = bank_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Categorising qualitative features (categorical columns)
categorical_features = bank_df.select_dtypes(include=['object', 'bool']).columns.tolist()


# In[19]:


numerical_features


# In[20]:


categorical_features
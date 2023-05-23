#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns

# Load dataset
heart_disease = pd.read_csv('heart1.csv')

# Create correlation matrix
corr_matrix = heart_disease.corr()

# Create heatmap
sns.heatmap(corr_matrix, annot=True)

# Create pair plot
sns.pairplot(heart_disease)


# In[ ]:





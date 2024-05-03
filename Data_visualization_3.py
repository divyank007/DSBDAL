# %% [markdown]
# # 6. Data Visualization III
# Download the Iris flower dataset or any other dataset into a DataFrame. (e.g.,
# https://archive.ics.uci.edu/ml/datasets/Iris).
# Scan the dataset and give the inference as:
# 1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
# 2. Create a histogram for each feature in the dataset to illustrate the feature distributions.
# 3. Create a box plot for each feature in the dataset. Compare distributions and identify
# outliers.

# %%
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_theme( rc={ "figure.figsize": (4,3) } )

# %%
ds = pd.read_csv( "iris.csv" )
ds.head()

# %% [markdown]
# ## 1. Feature Information

# %% [markdown]
# Numeric features: `sepal_width`, `sepal_length`, `petal_length` and `petal_width`
# 
# Nominal features: `species`

# %%
ds.dtypes

# %%
ds.describe()

# %% [markdown]
# ## 2. Feature Distribution with Histograms

# %%
sns.histplot( data=ds , x="sepal_length" , bins=10 )

# %%
sns.histplot( data=ds , x="sepal_width" , bins=10 )

# %%
sns.histplot( data=ds , x="petal_length" , bins=10 )

# %%
sns.histplot( data=ds , x="petal_width" , bins=10 )

# %% [markdown]
# ## 3. Visualizing feature distribution with box plots

# %% [markdown]
# ### 3.1. Visualizing the distribution of numeric features

# %%
sns.boxplot( data=ds.drop( [ "species" ] , axis=1 ) )

# %% [markdown]
# ### 3.2. Visualizing the distribution of each numeric feature against `spec

# %%
sns.boxplot( data=ds , x="species" , y="sepal_length" )

# %%
sns.boxplot( data=ds , x="species" , y="sepal_width" )

# %%
sns.boxplot( data=ds , x="species" , y="petal_length" )

# %%
sns.boxplot( data=ds , x="species" , y="petal_width" )

# %% [markdown]
# ### 3.3. Comparing distributions

# %% [markdown]
# #### 3.3.1. Comparing distributions visually with KDE plots

# %%
# The sns.kdeplot() function in Seaborn is used to plot the Kernel Density Estimate (KDE) of a 
sns.kdeplot( ds , x="sepal_length" )

# %%
sns.kdeplot( ds , x="sepal_width" )

# %%
sns.kdeplot( ds , x="petal_length" )

# %%
sns.kdeplot( ds , x="petal_width" )

# %% [markdown]
# #### 3.3.2. Comparing distributions with Pearson's correlation

# %%
ds[ [ "sepal_length" , "sepal_width" , "petal_length" , "petal_width" ] ].corr()



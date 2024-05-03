# %% [markdown]
# ### Importing Libraries

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import set_theme

# %% [markdown]
# ### Checking available datasets

# %%
sns.get_dataset_names()

# %% [markdown]
# ### Load Titanic dataset 

# %%
titanic = sns.load_dataset('titanic')

# %%
titanic.shape

# %%
titanic.describe()

# %%
titanic.head()

# %%
set_theme(style="darkgrid")

# %% [markdown]
# ### Count Plots - Number of Passengers in Each Class

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='class', hue='class', data=titanic, palette='Set1', legend=False)
plt.title('Number of Passengers in Each Class')
plt.show()

# %% [markdown]
# ### Count Plot - Number of Passengers in Each Class Based on Gender 

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='class', hue='sex', data=titanic, palette='viridis')
plt.title('Number of Passengers in Each Class Based on Gender')
plt.show()


# %% [markdown]
# ### Box Plot - Age Distribution in Each Class

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='age', hue='class', legend=False, data=titanic, palette='coolwarm')
plt.title('Age Distribution in Each Class')
plt.show()


# %% [markdown]
# ### Count Plot - Number of Passengers who Survived (1) and Did Not Survive (0)

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='survived', data=titanic, palette='pastel')
plt.title('Number of Passengers who Survived (1) and Did Not Survive (0)')
plt.show()

# %% [markdown]
# ### Count Plot - Survival Status Based on Class 

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='survived', hue='class', data=titanic, palette='Set2')
plt.title('Survival Status Based on Class')
plt.show()

# %% [markdown]
# ### Histogram - Distribution of Ticket Prices

# %%
plt.figure(figsize=(10, 6))
sns.histplot(titanic['fare'], bins=30, kde=True, color='red')
plt.title('Distribution of Ticket Prices')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# The histogram shows the distribution of ticket prices paid by passengers aboard the Titanic. Most fares were concentrated at lower values, with a long tail of higher fares. This suggests a diverse range of passengers with varying economic means, with the majority paying lower fares.

# %% [markdown]
# Conclusion 
# 1. **Passenger Class Distribution:** Most people were in the cheapest class, and fewer were in the more expensive ones.
#   
# 2. **Passenger Class Distribution by Gender:** There were more men than women, especially in the cheapest class.
#   
# 3. **Distribution of Ticket Prices:** Most tickets were cheap, but some were really expensive, showing that people with different amounts of money were on the ship.


****************************************************************************************************


# %% [markdown]
# ### Import required libraries

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import set_theme
import numpy as np

# %% [markdown]
# ### Load dataset titanic into dataframe

# %%
titanic = sns.load_dataset('titanic')

# %% [markdown]
# ### Data Cleaning

# %%
titanic = titanic.dropna(subset=['age'])
df = titanic

# %%
titanic.isnull().sum()

# %%
Q1=df['age'].quantile(0.25)
Q3=df['age'].quantile(0.75)
IQR=Q3-Q1

# %%
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

lower_whisker = Q1 - 1.5*IQR
upper_whisker = Q3 + 1.5*IQR
df = df[(df['age'] > lower_whisker) & (df['age'] < upper_whisker)]

# %% [markdown]
# ### Boxplot

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=df)
plt.title('Age distribution by Gender and Survival')
plt.show()

# %% [markdown]
# ### Conclusion :

# %% [markdown]
# 1. Out of all the passangers on Titanic, there were more female passangers survived than male passangers.

# %% [markdown]
# 2. Most of the survived passangers are of middle age.



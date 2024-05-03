import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score

df = pd.read_csv("Social_Network_Ads.csv")
df.head()


df.isnull().sum()




df.describe()

df.dtypes


sns.countplot(x='Purchased', data=df)



sns.countplot(x='Purchased', data=df, hue='Gender')



df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})


df.head()


df.drop('User ID', axis=1, inplace=True)


df.head()


sns.heatmap(df.corr())


df.corr()


sns.boxplot(x='Purchased', y='Age', data=df)


from sklearn.linear_model import LogisticRegression

X = df[['Age']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


cm = confusion_matrix(y_test, y_pred)

tn,fp,fn,tp =cm.ravel()

a=accuracy_score(y_test,y_pred)
e=1-a
p=precision_score(y_test,y_pred)
r=recall_score(y_test,y_pred)

accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0]+ cm[1,1])
precision = cm[0,0] / (cm[0,0] + cm[0,1])
error_rate = (cm[0,1] + cm[1,0]) / (cm[0,0] + cm[0,1] + cm[1,0]+ cm[1,1])
recall = cm[0,0] / (cm[0,0]+cm[1,0])
print(accuracy)
print(precision)
print(error_rate)
print(recall)


print(cm)


x_vals = np.linspace(X_test['Age'].min(), X_test['Age'].max(), 70)
y_vals = clf.predict_proba(x_vals.reshape(-1, 1))[:, 1]

plt.figure(figsize=(16,6))
sns.scatterplot(x=x_vals, y=y_vals)


plt.figure(figsize=(16, 6))
sns.countplot(x="Age", hue="Purchased", data=df) 

# %% [markdown]
# ## Analysis
# 1) The prediction graph of y_pred(probability) vs X_test showing the Sigmoid curve
# 2) The 'Age' is the distinguishable feature having more correlation with 'Purchased' feature.

# %%

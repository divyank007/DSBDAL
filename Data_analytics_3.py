import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report

# %%
df = pd.read_csv("Iris.csv")
df.head()

# %%
df.isnull().sum()

# %%
df = df.dropna()
df.dtypes

# %%
def remove_outliers(df, column, species):
    subset_df = df[df['Species'] == species]
    Q1 = subset_df[column].quantile(0.25)
    Q3 = subset_df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df.loc[(df['Species'] == species) & ((df[column] < lower_bound) | (df[column] > upper_bound)), column] = None

    return df


# %%
df = remove_outliers(df, 'PetalWidthCm', 'Iris-setosa')
df = remove_outliers(df, 'SepalWidthCm', 'Iris-virginica')
df = remove_outliers(df, 'PetalLengthCm', 'Iris-setosa')

# %%
X = df.drop('Species', axis=1)
y = df['Species']  


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# %%
cm = confusion_matrix(y_test, y_pred)
print(cm)

plot_confusion_matrix(conf_mat=cm, figsize=(5,5), show_normed=True)
# plot_confusion_matrix(conf_mat=cm, figsize=(5,5), show_normed=True)
plt.show()

accuracy = np.trace(cm) / np.sum(cm)
print("Accuracy:", accuracy)

precision_0 = cm[0, 0] / np.sum(cm[0, :])
print("Precision (Class 0):", precision_0)

precision_1 = cm[1, 1] / np.sum(cm[1, :])
print("Precision (Class 1):", precision_1)

precision_2 = cm[2, 2] / np.sum(cm[2, :])
print("Precision (Class 2):", precision_2)

error_rate = 1 - accuracy
print("Error Rate:", error_rate)

recall_0 = cm[0, 0] / np.sum(cm[:, 0])
print("Recall (Class 0):", recall_0)

recall_1 = cm[1, 1] / np.sum(cm[:, 1])
print("Recall (Class 1):", recall_1)

recall_2 = cm[2, 2] / np.sum(cm[:, 2])
print("Recall (Class 2):", recall_2)


# %%

report = classification_report(y_test, y_pred)
print(report)



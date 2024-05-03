import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression

df = pd.read_csv('HousingData.csv')
df.head(10)

df.dtypes

df.isnull().sum()

df = df.fillna(df.mean())

def get_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] <= lower_bound) | (df[column] >= upper_bound)][column].tolist()
    return outliers
    
print(get_outliers(df,'MEDV'))

print(get_outliers(df,'CRIM'))

zscore = np.abs(stats.zscore(df))
zscore

df1 = df[(zscore < 4).all(axis=1)]

df1.shape

df.corr()
sns.heatmap(df.corr())

X = pd.DataFrame(np.c_[df1['LSTAT']], columns = ['LSTAT'])
Y = df1['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

from sklearn.metrics import mean_squared_error,r2_score

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

b1 = model.coef_[0]
b0 = model.intercept_

print(b1)
print(b0)

xi = df1['RM']
yi = df1['MEDV']
xi_mean = np.mean(xi)
yi_mean = np.mean(yi)

pro_mean_diff = np.sum((xi - xi_mean) * (yi - yi_mean))
mean_diff_sq = np.sum((xi - xi_mean)**2)

b1 = pro_mean_diff / mean_diff_sq
print(b1)

b0 = yi_mean - (b1 * xi_mean)
print(b0)

from sklearn.preprocessing import MinMaxScaler
normalize = ['LSTAT', 'RM']
scaler = MinMaxScaler()
df1.loc[:, normalize] = scaler.fit_transform(df1[normalize])

X1 = pd.DataFrame(np.c_[df1['LSTAT'], df1['RM']], columns = ['LSTAT','RM'])
Y1 = df1['MEDV']
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size = 0.2, random_state=5)
lin_model.fit(X_train1, Y_train1)

y_pred = lin_model.predict(X_test1)
mse = mean_squared_error(Y_test1, y_pred)
r2 = r2_score(Y_test1, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

b1 = lin_model.coef_[0]
b0 = lin_model.intercept_

print(b1)
print(b0)


plt.scatter(Y_test1, y_pred)

ideal_values = np.linspace(min(Y_test1), max(Y_test1))
plt.plot(ideal_values, ideal_values, color='red')

plt.xlabel("Price: in $1000's")
plt.ylabel("Predicted value")
plt.title("True value vs predicted value : Linear Regression")
plt.show()
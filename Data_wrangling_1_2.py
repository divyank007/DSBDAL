# %%
import numpy as np
import pandas as pd

# %%
data = pd.read_csv("data.csv")

# %%
data.head()

# %%
data.isnull()

# %%
data.isnull().sum()

# %%
data['Car'].fillna(data['Car'].mean(),inplace=True)

data['Car'].fillna(method='ffill' inplace=True)
data.replace('?',0,inplace=True)

# %%
data.isnull().sum()

# %%
data['BuildingArea'].fillna(data['BuildingArea'].mean(),inplace=True)
data['YearBuilt'].fillna(data['YearBuilt'].mean(),inplace=True)

# %%
data.isnull().sum()

# %%
data.dtypes

# %%
data['CouncilArea'].fillna(data['CouncilArea'].mode()[0],inplace=True)

# %%
data.isnull().sum()

# %%
data.dtypes

# %%
data['Car']=data['Car'].astype('int')

# %%
data.dtypes

# %%
data['Postcode'] = data['Postcode'].astype('int64')
data['Bedroom2'] = data['Bedroom2'].astype('int64')
data['Bathroom'] = data['Bathroom'].astype('int64')
data['Car'] = data['Car'].astype('int64')
data['YearBuilt'] = data['YearBuilt'].astype('int64')

# %%
data.dtypes

data['Gender'].replace({"male":0,"female":1})
# %%

sns.countplot(x='Suburb' data=data)
data['Suburb'].unique()

//Label encoding and one hot encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Suburb']=le.fit_transform(data['Suburb'])
data

# %%
data['CouncilArea'].unique()

# %%
data['CouncilArea']=le.fit_transform(data['CouncilArea'])

# %%
data

# %%
data=pd.get_dummies(data,columns=['Distance'])

# %%
data

# %%
from sklearn.preprocessing import OneHotEncoder
v=OneHotEncoder(sparse=False)
v.fit(data)
v2=v.transform(data)
print(v2)

# %%
s=data['SellerG']
s1=pd.get_dummies(s,prefix="SELLERG")
s=pd.concat([s,s1],axis=1)


# %%
print(s)

***********************************************************************************************************************



# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df = pd.read_csv("./StudentsPerformance.csv")
df.head()

# %%
df.dtypes

# %%
df.info()

# %%
df.describe()
# math score is an object so we need to convert its type to float64

# %%
df.isnull().sum()

# %%
df.shape

# %%
df.isnull()

# %%
df.head()

# %%
cols = ['gender', 'group', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'math_score', 'reading_score', 'writing_score']
df.columns = cols

# %%
df.head()

# %%
df['math_score'].unique()

# %%
df['math_score'] = df['math_score'].replace('?', float('nan'))

# %%
df['math_score'].unique()

# %%
df['math_score'] = df['math_score'].astype('float64')

# %%
df.dtypes

# %%
df.describe()

# %%
sns.boxplot(data = df, x = 'reading_score')

# %%
sns.boxplot(data = df, x = 'writing_score')

# %%
df.isnull().sum()

# %%
mean_math_score = df['math_score'].dropna().mean()
print(mean_math_score)
df['math_score'] = df['math_score'].fillna(mean_math_score)

# %%
df.isnull().sum()

# %%
mean_reading_score = df['reading_score'].dropna().mean()
print(mean_reading_score)
df['reading_score'] = df['reading_score'].fillna(mean_reading_score)

# %%
mean_writing_score = df['writing_score'].dropna().mean()
print(mean_writing_score)
df['writing_score'] = df['writing_score'].fillna(mean_writing_score)

# %%
df.isnull().sum()

# %%
sns.boxplot(data = df, x = 'math_score')

# %%
sns.boxplot(data = df, x = 'reading_score')

# %%
sns.boxplot(data = df, x = 'writing_score')

# %%
# data normalization
def min_max_normalize(name):
    global df
    df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())

min_max_normalize('math_score')
min_max_normalize('reading_score')
min_max_normalize('writing_score')

# %%
df.head(10)

# %%
df

# %%
gender_map = {'male' : 1, 'female' : 0}
df['gender'] = df['gender'].map(gender_map)
df

# %%
df.dtypes

# %%
lunch_map = {"standard" : 1, "free/reduced" : 0}
df['lunch'] = df['lunch'].map(lunch_map)
df

# %%
df

# %%
df.dtypes

# %%
def encode_categorical(feature):
    label = 0
    values = df[feature].unique()
    for val in values:
        df.loc[df[feature] == val, feature] = label
        label+=1
    df[feature] = df[feature].astype('int64')

encode_categorical('group')
encode_categorical('parental_level_of_education')
df

# %%
df.dtypes

# %%
test_map = {'none' : 0, 'completed' : 1}
df['test_preparation_course'] = df['test_preparation_course'].map(test_map)
df

# %%
df.dtypes

# %%
sns.boxplot(data = df, x = 'math_score')

# %%
def remove_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[feature] >= lower_bound) & (df[feature] <=upper_bound)]

    return df
df = remove_outliers(df, 'math_score')

# %%
sns.boxplot(data = df, x = 'math_score')

# %%
from sklearn.preprocessing import StandardScalar
scalar = StandardScalar()
# min_max_scalar = MinMaxScalar()

df['math_zscore'] = scalar.fit_transform(df['math_score'])
df

# %%





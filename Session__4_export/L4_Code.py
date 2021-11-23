#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Dr Ekaterina Abramova
Lecture 4 Numpy; Pandas; Sklearn
Reference: Jake VanderPlas ("Python for Data Science Handbook")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
# Check sklearn version 
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#%% BROADCASTING
a = np.arange(5)
a * 4 # scalar value 4 has been broadcast to all of the elements in a

x = np.arange(12).reshape(3,4)
y = np.arange(12).reshape(3,4)
x * y
'''
array([[  0,   1,   4,   9],
       [ 16,  25,  36,  49],
       [ 64,  81, 100, 121]])
'''
x = np.ones((3,4), dtype = np.int) 
'''
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
'''
y = np.arange(1,5) # array([1, 2, 3, 4])
z = np.arange(1,4) # array([1, 2, 3])
# smaller array broadcast down columns
x - y              # (3,4) (4,)
x - y.reshape(1,4) # (3,4) (1,4) dim 4 and 4 match; one of dims is 1 so ok
# smaller array broadcast across rows
x - z.reshape(3,1) 
# next won't work
x - y.reshape(4,1) # (3,4) (4,1) dimensions don't match 3 and 4; 4 and 1
x - z              # (3,4) (3,)

x = np.ones((3,3), dtype = np.int) 
z = np.array([1,2,3])


np.random.seed(1234)
a = np.random.randn(4, 3)
mu = a.mean(axis = 0) # column-wise mean
# array([-0.30607832, -0.34951305,  0.83187808])
mu.shape # (3,)
demeaned = a - mu # (4,3) (3,)
demeaned2 = a - mu.reshape(1, 3) # (4,3) (1,3) 


#%% SLICING NUMPY ARRAYS
a = np.arange(10,16) # array([0, 1, 2, 3, 4, 5]) i.e. just a vector of 6 numbers
a[0]     # 0  i.e. element in 1st position
a[2]     # 2  i.e. element in 3rd position
a[2:5]   # array([2, 3, 4]) slice of the array 
a[1:4:2] # array([1, 3]) i.e. start at index 1, end at index 4, in steps of 2
a[3:5] = 0
a

a = np.arange(9, dtype = np.int).reshape(3,3)
"""
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
"""
a[0, 0] # 0

a[1, :] # note we selected entire second row, this is equivalent to:
a[1, ]
a[1]

s = a[1:3, 1:3] # array([[4, 5],
#                       [7, 8]])
s[:] = 0
s
a
'''
array([[0, 1, 2],
       [3, 0, 0],
       [6, 0, 0]])
'''

# 6 by 6 array
a = np.zeros(36).reshape(6, 6)
a[0,] = list(range(6))
a[1,] = list(range(10, 16))
a[2,] = list(range(20, 26))
a[3,] = list(range(30, 36))
a[4,] = list(range(40, 46))
a[5,] = list(range(50, 56))

# execute single line at a time and observe the output
print(a)
a[0]         # array([0., 1., 2., 3., 4., 5.])
a[0, 3:5]    # array([3., 4.])
a[4: , 4:]   # array([[44., 45.], [54., 55.]])
a[: , 2]     # array([ 2., 12., 22., 32., 42., 52.])
a[2::2, ::2] # array([[20., 22., 24.], [40., 42., 44.]])

#%%INDEXING WITH BOOLEAN ARRAY
gender = np.array(["Female", "Male", "Male", "Male", "Female"]) 
data = np.array([4, 8, 3, 9, 2])
bools = gender == "Male"
s = data[bools] # selection from array data is obtained 
selection = data[gender == "Male"]

# select logical opposite of bools (i.e.)
~ bools # array([True, False, False, False, True])
s = data[~ bools]         

data[gender == "Male"] = 0 
data # array([4, 0, 0, 0, 2])



#%% PANDAS
# SERIES ----------------------------------------------------------------------
s = pd.Series(np.array([1,2,3]), index=["a", "b", "c"])
s = pd.Series([1,2,3], index=["a", "b", "c"])
s.values
s.index

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
s = pd.Series(sdata)
s.name = 'population'
s.index.name = 'state'

#indexing
s[0]
s['a']

#slicing
s[:] # equivaletnly s[0:3]

s[0:2] # start:stop, where we finish at stop-1
#a 1
#b 2

s['a':'c'] # start:stop, where end-point is INCLUSIVE
"""
a    1
b    2
c    3
"""

# all Numpy Universal Functions can be used on Series
s2 = np.add(s, 10)
"""
a    11
b    12
c    13
dtype: int64
"""

b = np.isfinite(s)
"""
a    True
b    True
c    True
dtype: bool
"""

# NumPy like operations on Series:
s + s
s * 2
s > 0
'a' in s

# check if data contains NaN
s3 = pd.Series([None, 1, 2])
"""
0    NaN
1    1.0
2    2.0
dtype: float64
"""
pd.isnull(s3)  # top-level pandas function to check for NaN
s3.isnull()    # series instance method 
np.isnan(s3)   # can use ufunc of numpy on Series
"""
0     True
1    False
2    False
dtype: bool
"""
s3.isnull().any() # True
s3.isnull().sum() # 1


# DATAFRAMES -------------------------------------------------------------------
#%% DATAFRAME 1
data = {'GDP' : [3.8, 3.7, 3.2],
        'CPI' : [1.6, 3, 2.1],
		'year': [2019, 2020, 2021]
        } 

df = pd.DataFrame(data, columns=['year','GDP','CPI','result']) # row index remained unnamed.
"""
   year  GDP  CPI result
0  2019  3.8  1.6    NaN
1  2020  3.7  3.0    NaN
2  2021  3.2  2.1    NaN
"""

df.values
"""
array([[2019, 3.8, 1.6, nan],
       [2020, 3.7, 3.0, nan],
       [2021, 3.2, 2.1, nan]], dtype=object)
"""
df.index
"""
RangeIndex(start=0, stop=3, step=1)
"""

df.dtypes
'''
year        int64
GDP       float64
CPI       float64
result     object
dtype: object
'''

# Name rows
df = pd.DataFrame(data, 
                  index = ['row1', 'row2', 'row3'], 
                  columns = ['year','GDP','CPI','result']) 
df.index.name = "obs"

# intialise 2nd dataframe to the size of the 1st one:
df2 = pd.DataFrame(np.nan, index = df.index, columns = df.columns)  
df3 = pd.DataFrame(np.nan, index = np.arange(3), columns = df.columns)
df4 = pd.DataFrame(data) # just pass the data in to covert

# retrieve columns as Series using:
# a) dict-like notation
df['CPI']
# b) by attribute
df.CPI
"""
0    1.6
1    3.0
2    2.1
Name: CPI, dtype: float64
"""
type(df.CPI) # pandas.core.series.Series (retrieved a Series)


# create new column
df['new'] = 0

# delete row
df.drop([0])
"""
   year  GDP  CPI result
1  2017  3.7  3.0    NaN
2  2018  3.2  2.1    NaN
"""
print(df)        # original object unchanged

# delete column
# specify that 'new' label is to be searched along the column axis
newD = df.drop(['new'], axis=1) 
print(newD)
print(df)        # original object unchanged


# Many functions have the option to manipulate object in-place (affecting the 
# original dataframe directly and not returning a new object). 
df.drop([0], inplace=True) 
print(df)        # original object changed

del df['result'] # using dict-like notation for column deletion
print(df)        # original object changed


#%% DATAFRAME 2
data = {'GDP' : [3.8, 3.7, 3.2],
        'CPI' : [1.6, 3, 2.1],
		'year': [2019, 2020, 2021]
        } 

df = pd.DataFrame(data, 
                  index = ['row1', 'row2', 'row3'],
                  columns=['year','GDP','CPI','result'])

"""
      year  GDP  CPI result
row1  2016  3.8  1.6    NaN
row2  2017  3.7  3.0    NaN
row3  2018  3.2  2.1    NaN
"""

# start:stop
df[0:2] # row selection
"""
      year  GDP  CPI result
row1  2016  3.8  1.6    NaN
row2  2017  3.7  3.0    NaN
"""

# 'label'
df['CPI'] # column selection (Series)
df.CPI    # column selection (Series)

# 1 or more columns
df[['CPI','GDP']]  # index using column index values (result based on order of index label)
"""
   CPI  GDP
0  1.6  3.8
1  3.0  3.7
2  2.1  3.2
"""


# loc[val] single row
df.loc['row1']
"""
year      2016
GDP        3.8
CPI        1.6
result     NaN
Name: row1, dtype: object
"""
type(df.loc['row1']) # extracted row is a Series pandas.core.series.Series

# loc[val] subset of rows
df.loc['row1':'row3']
"""
      year  GDP  CPI result
row1  2016  3.8  1.6    NaN
row2  2017  3.7  3.0    NaN
row3  2018  3.2  2.1    NaN
"""
df.loc[['row3','row1']]

# loc[:,val] single column
df.loc[:,'CPI']
"""
row1    1.6
row2    3.0
row3    2.1
Name: CPI, dtype: float64
"""

# loc[:,val] subset of columns
df.loc[:,'year':'CPI']
"""
      year  GDP  CPI
row1  2016  3.8  1.6
row2  2017  3.7  3.0
row3  2018  3.2  2.1
"""

df.loc[:,['result','CPI']]
"""
     result  CPI
row1    NaN  1.6
row2    NaN  3.0
row3    NaN  2.1
"""


# loc[val1, val2]
df.loc['row2', 'GDP'] # 3.7
df.loc['row1':'row3', 'year':'result']
"""
      year  GDP  CPI result
row1  2016  3.8  1.6    NaN
row2  2017  3.7  3.0    NaN
row3  2018  3.2  2.1    NaN
"""


# iloc[where]
df.iloc[0]
"""
year      2016
GDP        3.8
CPI        1.6
result     NaN
Name: row1, dtype: object
"""
df.iloc[0:3] 
"""
      year  GDP  CPI result
row1  2016  3.8  1.6    NaN
row2  2017  3.7  3.0    NaN
row3  2018  3.2  2.1    NaN
"""
df.iloc[[2,0]]
"""
      year  GDP  CPI result
row3  2018  3.2  2.1    NaN
row1  2016  3.8  1.6    NaN
"""

# iloc[:, where]
df.iloc[:, 2]      # CPI column
df.iloc[:, 0:2]    # year, GDP column
df.iloc[:, [3,2]]  # result, CPI


# DATAFRAME METHODS
data = {'GDP' : [3.8, 3.7, 3.2],
        'CPI' : [1.6, 3, 2.1],
		'year': [2016, 2017, 2018]
        } 
df = pd.DataFrame(data, columns=['year','CPI'])

df['tAhead'] = df.CPI.shift(-1)
df['tLag'] = df.CPI.shift(1)


#%% --- DETECTING MISSING DATA -------------------------------------------------:
# --- numpy -------------------------------------------------------------------
# If we insert a None type this presents and issue for when we check for NaNs:
a = np.array([None, 1, 2]) # array([None, 1, 2], dtype=object)
np.isnan(a)                # TypeError (cannot deal with None)

# The following will work as we use np.nan which is treated as a float:
a = np.array([np.nan, 1, 2]) # array([nan, 1, 2], dtype=object)
np.isnan(a)                  # array([ True, False, False])
np.isnan(a).sum()            # 1

# --- Series ------------------------------------------------------------------
a = np.array([np.nan, 1, 2])
s = pd.Series(a)
s.isnull()
'''
0     True
1    False
2    False
dtype: bool
'''
s.isnull().sum() # 1
s.isnull().any() # True

# --- DataFrame ---------------------------------------------------------------
data = {'GDP' : [3.8, 3.7, 3.2],
        'CPI' : [1.6, 3, 2.1],
		'year': [2019, 2020, 2021]} 
df = pd.DataFrame(data, columns=['year','GDP','CPI','result'])
# Boolean array
df.isnull()
'''
    year    GDP    CPI  result
0  False  False  False    True
1  False  False  False    True
2  False  False  False    True
'''
# Total number of NaNs in each column
df.isnull().sum()
'''
year      0
GDP       0
CPI       0
result    3
dtype: int64
'''
# Single number indicating number of NaNs in entire dataframe
df.isnull().values.sum() # 3


# --- FILLING IN MISSING DATA ------------------------------------------------:
data = {'gender': ["male", "female", "male", "female", "male", "female"],
        'age' : [3.3, 3.7, 3.2, 4.1, 3.3, 1.6],
        'x' : [1.6, np.nan, 2.1, np.nan, np.nan, 2.2]
		} 
df = pd.DataFrame(data)
'''
   gender  age    x
0    male  3.3  1.6
1  female  3.7  NaN
2    male  3.2  2.1
3  female  4.1  NaN
4    male  3.3  NaN
5  female  1.6  2.2
'''

df['x'].fillna() # will not work as need to specify the value

df['x'].fillna(value = 0)
'''
0    1.6
1    0.0
2    2.1
3    0.0
4    0.0
5    2.2
Name: x, dtype: float64
'''

df['x'].fillna(method = 'ffill') # same if used 'pad'
'''
0    1.6
1    1.6
2    2.1
3    2.1
4    2.1
5    2.2
Name: x, dtype: float64
'''

df['x'].fillna(method = 'bfill') # same if used 'backfill'
'''
0    1.6
1    2.1
2    2.1
3    2.2
4    2.2
5    2.2
Name: x, dtype: float64
'''


# Let's fill in missing values in 1 column based on another:  
gp = df.groupby('gender')
val = gp.transform('median').age
'''
0    3.3
1    3.7
2    3.3
3    3.7
4    3.3
5    3.7
'''
df['x'].fillna(val, inplace=False)
'''
0    1.6
1    3.7
2    2.1
3    3.7
4    3.3
5    2.2
Name: x, dtype: float64
'''

#%% ----------------------------- SKLEARN -------------------------------------
# Data Representation 
# OPTION 1: use seaborn library to load the dataset ---------------------------
iris = sns.load_dataset('iris') # read-in as DataFrame
iris.head()
'''
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
'''

sns.set() # set theme
sns.pairplot(iris, hue = 'species', height = 1.5)

X_iris = iris.drop('species', axis = 1) # NOT in-place (iris is unchanged)
X_iris.shape # (150,4)

y_iris = iris['species']
y_iris.shape # (150,)


# OPTION 2: use sklearn library to load the dataset ---------------------------
iris = load_iris()

type(iris)# sklearn.utils.Bunch
# Bunch object, which is very similar to dictionary

iris.keys() 
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

iris['target_names'] # array of strings
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
# i.e.    0         1             2

iris['feature_names'] # list of strings
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

iris['data'].shape # (150, 4) i.e. number of samples x number of features

iris['data'][0:5]
'''
array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       [5. , 3.6, 1.4, 0.2]])
Note all petal width is 0.2!!!
'''

iris['target']
'''
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
Note target varaibles are ordered as 0, 1 then 2, and not random.
'''


#%% Regression Toy Example
np.random.seed(42)
x = 10 * np.random.rand(50)
y = 2 * x - 1 + np.random.randn(50)
plt.scatter(x, y)
plt.show()

model = LinearRegression() # remember by default fit_intercept=True

# Reshape 1D array into matrix [n_samples, n_features], shape should be (50,1)
x.shape # (50,)
n = len(x)
X = x.reshape(n, 1) # make X a single column matrix of data 
X.shape # (50, 1)

model.fit(X, y)

model.coef_ # array with results: array([1.9776566])
model.intercept_ # single value: -0.9033107255311164
 
# Predict new points
# numpy.linspace(start, stop, num=50, endpoint=True)
x_test = np.linspace(-1, 11) # note default number of numbers is 50
X_test = x_test.reshape(n, 1) # (50,) -> (50,1)
y_pred = model.predict(X_test)  # predict y based on new X and our model
# highlight next 3 lines together and run
plt.scatter(x, y)
plt.plot(x_test, y_pred)
plt.show()

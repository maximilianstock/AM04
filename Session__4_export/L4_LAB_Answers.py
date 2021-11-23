#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author : Dr Ekaterina Abramova
Lecture 4 LAB Answers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#%% Q1 NUMPY LAB
# c) --------------------------------------------------------------------------
np.random.seed(1234) # fix the random number generator
'''
Let's leverage the vectorization offered by numpy library. Your task is to 
simulate many random walks at once, e.g. 5000 random walks of 1000 steps each. 
This is just a small adjustment to the code from L3 Q5b.
If you have not done so yet, complete L3 Q5 first.
'''
nwalks = 5000
nsteps = 1000

# i) Adjust the draws variable you have written above to indicate the correct
# array size of draws so that to have 5000 rows and 1000 cols.
draws = np.random.randint(0, 2, size = (nwalks, nsteps))

steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(axis = 1) # obtain result across cols (across each walk)

# ii) Compute max and min values obained across all walks:
walks.max() # 122
walks.min() # -128

# iii) Let's check HOW MANY of the random walks reached over 20 (+ve or -ve). 
#      Step1: (as before) find boolean array of True, False for where element 
#             is above 20 in absolute value.
#      Step2: For each bool array check if any of the elements are True (result 
#             of such a check should produce single True/False value per walk).
#             Hint1: a.any() checks if >=1 elements are True in a bool array.
#             Hint2: walks are stored in rows (think of optional argument axis)
#      Step3: Find the sum across vector of True/False values from prev step
n = 20
hits20 = (np.abs(walks) >= n).any(axis = 1) # get bool array across cols
# array([ True,  True,  True, ..., False,  True, False])
# Obtain sum across True/False values to provide tot num of walks above 20.
hits20.sum() # remember True is treated as 1, therefore we can sum up the num
             # of occurances where walk reached + or - 20.
# 4723 walks crossed over 20

'''
Find the average number of steps that it took for the walks to cross the 
+/-20 value. Note: only the walks which actually reached + or - 20 should be
considered.  
'''
# i) Select only the walks where crossing happened (i.e. where hits20 is True):
crossed = walks[hits20] # 4723x1000

# ii) For each of those walks find boolean array of True and False values for  
#     where element is above + or - 20 in each walk.
crossed_bool = np.abs(crossed) >= n 

# iii) Find the location (step) where crossing happened for each walk.   
crossing_times = crossed_bool.argmax(axis = 1) # argmax() fill find the first occurance in the vector

# iv) Obtain the average crossing time
crossing_times.mean() # 345.7308913825958


# d) [EXTRA] ------------------------------------------------------------------
# --- Speed of execution ------------------------------------------------------
'''
Numpy’s functions are written in C, which results in them requiring less time &
less memory to perform operatins, than compared to built-in Python sequences. 
Equally the ability to perform operations on entire arrays without for loops 
makes them highly suitable for numerical data processing tasks.

Investigate the speed of computation using a list and a numpy array. Check
computation time in milliseconds of a multiplication by 2 of each element of a 
list and do the same for an array, when performing that computation 10 times. 

Initial variables are:
-	List containing as elements: numbers between 1 and 1,000,000
-	Numpy array containing numbers between 1 and 1,000,000 (stored as col vec)
             
There are 2 different ways of timing executions, 1) using time library and 2) 
using magic commands (see L4 Extra slides). Carry out timing using both ways.

Note: when solving this task for lists, use list comprehension for multiplying 
each element of a list by a value. 
'''

# i) Appraoch 1: write out a script which utilises the TIME LIBRARY
L = list(range(1000000))
a = np.arange(1000000)

# perform list timing
start = time.time()
# index variable is _ by tradition when ii is not used inside the for loop
# reminder: for loop should perform multiplication by 2 of each element of L
for _ in range(10): 
    newL  = [2*x for x in L]
end   = time.time()
t_L   = end-start
print("List time: ", t_L*1000, 'ms') # time in milliseconds

# perform numpy array timing
start2 = time.time()
# reminder: for loop should perform multiplication of array a by 2 
for _ in range(10):
    newa  = 2*a
end2 = time.time()
t_a  = end2-start2
print("Array time: ", t_a*1000, 'ms') # time in milliseconds


# ii) Approach 2: Perform timing using the console
'''
Magic command is a one-line operation which should be executed in Console (see 
L4 Extra Slides). You are allowed to write a 1 line for loop on a single line. 
For example if you had:

for ii in range(10):
    x = print(ii)

you could execute this in Console as:
for ii in range(10): x = print(ii)

Write a single line of code (one for list and one for arrays) using magic 
command and execute in console to report execusion time.
'''
# %time for _ in range(10): newL = [2*x for x in L]
# %time for _ in range(10): a2 = 2*a


#%% Q2
###############################################################################
#%% COIN TOSS GAME RULES -----------------------------------------------------:
'''
Win £2 if you get heads
Lose £1 if you get tails

SINGLE COIN TOSS (GAME 1)
Expected payoff:
    0.5*2 + 0.5*(−1) = 0.5
Std dev of payoff:
    sqrt( (0.5 * (2 − 0.5) ** 2) + (0.5 * (− 1 − 0.5) ** 2) ) = 1.5
    
TWO COIN TOSSES (GAME 2)
Expected payoff:
    0.5 * (1 + 1) + 2 * 0.25 * (1 − 0.5) + 0.25 * (− 0.5 − 0.5) = 0.5
Std dev of payoff:
    sqrt(0.25 * (2 − 0.5) ** 2 + 0.5 * (0.5 − 0.5) ** 2 + 0.25 * (−1 − 0.5)**2) 
    = 1.06
'''


#%% THIS SECTIONS AIMS TO SHOW HOW TO USE PYTHON AS A CALCULATOR ONLY --------:
# Simply transferring equations into code and using Python as a calculator.

# --- GAME 1: Single coin toss  
n = 1      # number of coin tosses
p = 0.5**n # pWin = pLoss for the toss

H = 2/1    # heads wins £2
T = - H/2  # tails loses -£1
# s = {H, T} set of all possible outcomes

payoff1 = p * H + p * T

std_PO_1 = (p * (H - payoff1) ** 2 + p * (T - payoff1) ** 2) ** 0.5 # sqrt

print("Payoff:", payoff1, "StdPayoff:", '%.2f' % std_PO_1)


# --- GAME 2: Two coin tosses 
n = 2      # number of coin tosses
p = 0.5**n # 0.25 pWin = pLoss on each toss

H = 2/n    #  £1   heads wins
T = - H/2  # -£0.5 tails loses
# s = {HH, TT, HT, TH} set of all possible outcomes

payoff2 = p * (H + H) + p * (H + T) + p * (T + H) + p * (T + T)

std_PO_2 = (p * ((H + H) - payoff2) ** 2 + \
           p * ((H + T) - payoff2) ** 2 + \
           p * ((T + H) - payoff2) ** 2 + \
           p * ((T + T) - payoff2) ** 2) ** 0.5 # sqrt
    
print("Payoff:", payoff2, "StdPayoff:", '%.2f' % std_PO_2)


#%% HOW TO USE FOR LOOPS TO PLAY GAME WITH N TOSSES --------------------------:
# Generalising above solution to n tosses. However note: since for loop are not
# as efficient ndarrays we will use only use n < 20.
'''
NOTE! There is a significant overhang for converting product output to other
data types, such as lists, arrays and dataframes. Therefore only go up to n=20.

This means that this operation is performed very fast:
    outcomes = product(s, repeat = n) 
But conversion to datatypes we can use such as list, is very slow for large n:
    list(outcomes)
'''

# --- FOR LOOPS SOLUTION
from itertools import product

n = 3        # number of tosses
p = 0.5**n   # pWin = pLoss on each toss

H = 2/n      #  £0.6667 heads wins
T = - H/2    # -£0.3333 tails loses
s = (H, T)   # set of all possible outcomes

outcomes = list(product(s, repeat = n)) # typecast to list to reveal results

print("Possible outcomes of experiment:")
for result in product("HT", repeat = n):
    print(result)
 
print("Corresponding probabilities:")
for result in outcomes:
    print(result)

# calculation of payoff
payoffN = 0
numOutcomes = len(outcomes)
for ii in range(numOutcomes):
    payoffN += p * sum(outcomes[ii])

# calculation of standard deviation of payoff
tot = 0
for ii in range(numOutcomes):
    # calcualte the squared sum
    tot += p * (sum(outcomes[ii]) - payoffN) ** 2
std_PO_N = tot ** 0.5 # sqrt

print("PayoffN:", '%.2f' % payoffN, "StdPayoff:", '%.2f' % std_PO_N)


#%% HOW TO USE NUMPY AND PANDAS LIBRARIES TO PLAY GAME WITH N TOSSES ---------:
# --- NUMPY AND PANDAS SOLUTION
from itertools import product
import numpy as np
import pandas as pd

n = 10      # number of tosses
p = 0.5**n   # pWin = pLoss on each toss
H = 2/n      #  £1   heads wins
T = - H/2    # -£0.5 tails loses
s = (H, T)   # set of all possible outcomes

outcomes = list(product(s, repeat = n)) # typecast to list to reveal results
# outcomes2 = 
# np.array(s)[np.rollaxis(np.indices((len(s),) * n), 0, n+1).reshape(-1, n)]

df = pd.DataFrame(outcomes) 
df.head()

# PAYOFF
df['sums'] = df.sum(axis = 1)
df.head()

payoff = np.sum(df['sums'] * p)


# STANDARD DEVIATION
df['sqDiff'] = (df['sums'] - payoff) ** 2
df.head()

stdDev = (np.sum(df['sqDiff'] * p)) ** 0.5 # sqrt

print("PayoffN:", '%.2f' % payoff, "StdPayoff:", '%.2f' % stdDev)


#%% HOW TO CREATE A FUNCTION FOR TESTING 1-N ITERATIVELY ---------------------:
# GENERAL GAME: n coin tosses NUMPY AND PANDAS, with function 
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Function definition. Highlight all function and run it, to place it in 
# memory and make avaialble for use.
def gameN(n):
    p = 0.5**n   # pWin = pLoss on each toss
    H = 2/n      #  £1   heads wins
    T = - H/2    # -£0.5 tails loses
    s = (H, T)   # set of all possible outcomes
    outcomes = list(product(s, repeat = n)) #typecast to list to reveal results
    #
    df = pd.DataFrame(outcomes) 
    # PAYOFF
    df['sums'] = df.sum(axis = 1)
    payoff = np.sum(df['sums'] * p)
    # STANDARD DEVIATION
    df['sqDiff'] = (df['sums'] - payoff) ** 2
    stdDev = (np.sum(df['sqDiff'] * p)) ** 0.5 # sqrt
    #
    return stdDev

N = 10 # max number of tosses allowed for our longest game
res = [0] * N
for ii in range(1, N+1): # must start at least from n = 1
    res[ii-1] = gameN(ii)
plt.plot(range(1,N+1), res, 'g-o') # x-axis numbering in human count ()



#%% Q3
# LOAD AND EXAMINE IRIS DATA --------------------------------------------------
iris = load_iris()
type(iris)            # sklearn.utils.Bunch
iris.keys()           # dict_keys(
#['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
iris['target_names']  # array of strings
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
# i.e.    0         1             2
iris['feature_names'] # list of strings ['sepal length (cm)', 
# 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
iris['data'].shape    # (150, 4) i.e. number of samples x number of features
iris['target'].shape  # ﻿(150,)


iris['data'][0:5, :]
'''
array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       [5. , 3.6, 1.4, 0.2]])
'''

iris['target']
# Note target varaibles are ordered as 0, 1 then 2, and not random.
'''
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
'''


# DATA PRE-PROCESSING AND VISUALISING -----------------------------------------
'''
Data is already all in cm therefore no need to standardize / normalise.

KNN uses all of the training data for decision making and saw above that target
vector corresponds to ordered data for classes 0, 1, 2. Therefore we need to 
shuffle observations randomly. This is automatically done inside the function
train_test_split(). 
'''

X_train, X_test, y_train, y_test = train_test_split(iris['data'], 
                                                    iris['target'], 
                                                    test_size = 0.25, # default 
                                                    random_state = 1234,
                                                    shuffle = True)   # default
# 112 x 4 train
# 38 x 4 test

# Typecast array to dataframe for plotting
df = pd.DataFrame(X_train, columns = iris['feature_names'])
# Add column with label for plotting
df['species'] = y_train
sns.pairplot(df, hue = "species", height = 1.5) 
# The plots show that 3 classes are well separated using sepal and petal data.


# BUILIDING MACHINE LEARNING MODELS -------------------------------------------
#K-NEAREST NEIGHBOUR K = 3
K = 3 # number of neighbours
# --- Create instance of the class, call it knn
knn = KNeighborsClassifier(n_neighbors = K)

# --- Fitting Model
knn.fit(X_train, y_train) # operates inplace

# --- Make Predictions for X_test
y_pred = knn.predict(X_test)
print(y_pred)
# [1 1 2 0 1 0 0 0 1 2 1 0 2 1 0 1 2 0 2 1 1 1 1 1 2 0 2 1 2 0 1 2 0 1 1 0 0 0]

# --- Model Accuracy
knn.score(X_test, y_test) # 0.9736842105263158
# all it does is the following:
np.mean(y_pred == y_test) # 0.9736842105263158
# We have a model with 97.4% accuracy on a test set

# --- Making Predictions
x_new = np.array([[5, 2.9, 1, 0.2]])
x_new.shape # (1, 4)

prediction = knn.predict(x_new)
print(prediction)
# 0 i.e. setosa

# Q: how do we know if we are correct?
# A: we don't know the correct species, that is the reason for building a model


# [EXTRA] KNN BY HAND ---------------------------------------------------------
n = len(X_test)
m = len(X_train)
y_pred = [0] * n # empty list
for ii in range(n):
    distVec = pd.Series(np.zeros(m)) # Pandas Series of size m (all 0s)
    a = X_test[ii, :] # row of test data (i.e. 1 test point)
    for jj in range(m):
        b = X_train[jj, :] # row of train data (1 train point)
        sqDiff = (a - b) ** 2
        distVec[jj] = np.sqrt(sqDiff.sum()) # distance to trainig point jj
    
    # Sort distance vector and corresponding training labels
    inds = distVec.argsort()
    distVec_sorted = distVec[inds]
    y_train_sorted = y_train[inds] # obtain y train in order of sorted dist vec
    
    # Obtain classification and compare to true answer
    firstK = y_train_sorted[0:K]
    unique, counts = np.unique(firstK, return_counts = True)
    y_pred[ii] = unique[counts.argmax()]

np.mean(y_pred == y_test) # 0.9736842105263158


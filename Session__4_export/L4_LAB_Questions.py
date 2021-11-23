#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author : Dr Ekaterina Abramova
Lecture 4 : LAB QUESTIONS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#%% Q1 NUMPY LAB
#  c) -------------------------------------------------------------------------
'''
Let's leverage the vectorization offered by numpy library again.
Your task is to simulate many random walks at once, e.g. 5000 random walks of 
1000 steps each. 
This is just a small adjustment to the code from L3 Q5b.
If you have not done so yet, complete L3 Q5 first.
'''
nwalks = 5000
nsteps = 1000

# i) Adjust the draws variable written below to produce the required / correct
# array size of draws so that to have 5000 rows and 1000 cols.
draws = np.random.randint(0, 2, size = steps)

steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(axis = 1) # obtain result across cols (across each walk)

# ii) Compute max and min values obained across all walks:


# iii) Let's check HOW MANY of the random walks reached over 20 (+ve or -ve). 
#      Step1: (as before) find boolean array of True, False for where element 
#             is above 20 in absolute value.
#      Step2: For each bool array check if any of the elements are True (result 
#             of such a check should produce single True/False value per walk).
#             Hint1: a.any() checks if >=1 elements are True in a bool array.
#             Hint2: walks are stored in rows (think of optional argument axis)
#      Step3: Find the sum across vector of True/False values from prev step
n = 20
hits20 = 
# Obtain sum across True/False values to provide tot num of walks above 20.


'''
Find the average number of steps that it took for the walks to cross the 
+/-20 value. Note: only the walks which actually reached + or - 20 should be
considered.  
'''
# i) Select only the walks where crossing happened (i.e. where hits20 is True):
crossed = 

# ii) For each of those walks find boolean array of True and False values for  
#     where element is above + or - 20 in each walk.
crossed_bool = 

# iii) Find the location (step) where crossing happened for each walk.
crossing_times = 

# iv) Obtain the average crossing time


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
L =   # initialise list
a =   # initialise array

# perform list timing
start = time.time()
# FOR LOOP GOES HERE: write for loop to perform list comprehension 10 times 
# reminder: for loop should perform multiplication by 2 of each element of L
end   = time.time()
t_L   = end-start
print("List time: ", t_L*1000, 'ms') # time in milliseconds                                                                                                                                                                                            seconds

# perform numpy array timing
start2 = time.time()
# FOR LOOP GOES HERE: write for loop to perform array multiplication 10 times 
# reminder: for loop should perform multiplication of array a by 2 
end2 = time.time()
t_a  = end2-start2
print("Array time: ", t_a*1000, 'ms') # time in milliseconds


# ii) Approach 2: Perform timing using the console
# %time for _ in range(10): newL = [2*x for x in L]
# %time for _ in range(10): a2 = 2*a


#%% Q2 COIN TOSS -------------------------------------------------------------:
'''
GAME RULES:
    
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
n = 1   # number of coin tosses
p =  # pWin = pLoss for the toss

H =     # heads wins £2
T =   # tails loses -£1
# s = (H, T) set of all possible outcomes

payoff = 

std_PO 

print("Payoff:", payoff, "StdPayoff:", '%.2f' % std_PO)


# --- GAME 2: Two coin tosses 
n = 2      # number of coin tosses
p =  # 0.25 pWin = pLoss on each toss

H =     #  £1   heads wins
T =   # -£0.5 tails loses
# s = {HH, TT, HT, TH} set of all possible outcomes

payoff2 = 

std_PO_2 = 
    
print("Payoff:", payoff2, "StdPayoff:", '%.2f' % std_PO_2)


#%% HOW TO USE FOR LOOPS TO PLAY GAME WITH N TOSSES --------------------------:
# Generalising above solution to n tosses. However note: since for loop are not
# as efficient as ndarrays we will use only use n < 20.
'''
NOTE! There is a significant overhang for converting product output to other
data types, such as lists, arrays and dataframes. Therefore only go up to n=20.

This means that this operation is performed very fast:
    outcomes = product(s, repeat = n) 
But conversion to datatypes we can use (such as list) is very slow for large n:
    list(outcomes)
'''

# --- FOR LOOPS SOLUTION
from itertools import product

n = 3  # number of tosses
p =    # pWin = pLoss on each toss

H =    #  £x heads wins
T =    # -£x tails loses
s = (H, T) # set of all possible outcomes

outcomes = list(product(s, repeat = n)) # typecast to list to reveal results

print("Possible outcomes of experiment:")
for result in product("HT", repeat = n):
    print(result)
 
print("Corresponding probabilities:")
for result in outcomes:
    print(result)

# calculation of payoff


# calculation of standard deviation of payoff


print("PayoffN:", '%.2f' % payoffN, "StdPayoff:", '%.2f' % std_PO_N)


#%% HOW TO USE NUMPY AND PANDAS LIBRARIES TO PLAY GAME WITH N TOSSES ---------:
# --- NUMPY AND PANDAS SOLUTION
from itertools import product
import numpy as np
import pandas as pd








#%% HOW TO CREATE A FUNCTION FOR TESTING 1-N ITERATIVELY ---------------------:
# GENERAL GAME: n coin tosses NUMPY AND PANDAS, with function 
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




#%% Q3 KNN
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


# b) Check the first 5 observations in the data (this should be a 5x4 array)


# c) Examine contents of target variable



# DATA PRE-PROCESSING AND VISUALISING -----------------------------------------

# d) 

# e) 

# f) Split data into train/test 75%/25% with random state 1234


# g) Typecast array to dataframe for plotting
df = pd.DataFrame(X_train, columns = iris['feature_names'])
# Add column with label for plotting
df['species'] = y_train
sns.pairplot(df, hue = "species", height = 1.5) 
# The plots show that 3 classes are well separated using sepal and petal data.


# K-NEAREST NEIGHBOUR K = 3 ---------------------------------------------------
K = 3 # number of neighbours

# --- Create instance of the class, call it knn
knn = 

# --- Fitting Model


# --- Make Predictions for X_test
y_pred = 
print(y_pred)

# --- Model Accuracy

# --- Making Predictions
x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = 
print(prediction)

# Q: how do we know if we are correct?
# A: 

# KNN BY HAND -----------------------------------------------------------------






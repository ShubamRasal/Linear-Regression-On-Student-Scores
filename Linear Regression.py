# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 23:29:36 2022

@author: Shubham
"""

#IMPORTING LIBRARIES:
#---------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os

os.chdir(r"D:\Data Science and Artificial Inteligence Semister- 1\SKLearn")
os.listdir()

df = pd.read_csv("student_scores.csv")

df.head(5)

df.shape

#statistical details:
#-----------------------
    
df.describe()

#--------------------------------------------------------------------------------

#Let's plot our data points on 2-D graph to eyeball our
#dataset and see if we can manually find any relationship
#between the data

df.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

#We can clearly see that there is a positive linear relation
#between the number of hours studied and percentage of score.
#--------------------------------------------------------------------------------

#Next step is to divide the data into "attributes" and "labels".
#Attributes- Independent variables
#Labels- Dependent variables
#Now we want too predict the percentage score depending upon the hours studied.
#Our attribute set will consits of the "Hours" column, and the label
#will be the "Score" column.

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

#The attributes are stored in the "X" variables
#We specified "-1" as the range for columns since we wanted our attribute set
#contain all the columns except the last one, which is "Scores". "Y" variables 
#contains the labels.

#--------------------------------------------------------------------------------

#The next step is to split this data into training and test sets.
#We'll do this by using Scikit-Learn's built-in train_test_split() method.

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

#The above script splits 80%  of the data to training set while 20% of the data to test set.
#the test_size variable is where we actually specify the proportion of test set.
#--------------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#To retrieve the intercept:
#-----------------------------------
    
print(regressor.intercept_)

#For retrieving the slope (coefficient of X):
#-----------------------------------------------
    
print(regressor.coef_)

#This means that for every one unit of change in hours studied, the change in
#the score is about 9.91%.
#If a student studies one hour more than they previously studied for an exam,
#they can except to achieve 
#an increase of 9.91% in the score achieved by the student previously.
#--------------------------------------------------------------------------------------

#It's time to make some predictions
#We will use our test data and see how accurately our algorithm predicts the
#percentage score.

y_pred = regressor.predict(X_test)

#the y_pred is a numpy array that contains all the predicted values in the
#input values in the X_test series.
#--------------------------------------------------------------------------------

#To compare the actual output values for X_test with the predicted values
#exectue the following script:

df =pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

#the predicted percentages are close to the actual ones.
#----------------------------------------------------------------------------------

#Evaluating the Algorithm:
#---------------------------
    
#The final step is to evaluate the performance of algorithm
#1. Mean Absolute Error is the mean of the absolute value of the errors
#2. Mean Squared Error is the mean of the squared errors.
#3. Roots Mean Squared Error is the square root of the mean of the squared errors.

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#The value of root mean squared error is 4.64, which is less than 10% of the
#mean of the mean value
#of the percentages of all the students i.e. 51.48.
#This means that our algorithm did a decent job.
#------------------------------------------------------------------------------------------
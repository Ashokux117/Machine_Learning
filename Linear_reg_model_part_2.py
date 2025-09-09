

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#load dataset
dataset = pd.read_csv(r"C:\Users\DELL\Downloads\Salary_Data.csv")

#  split dataset into x and y

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(x_train,y_train)

y_pred=regresor.predict(x_test)

comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison) 

plt.scatter(x_test,y_test, color="red")
plt.plot(x_train,regresor.predict(x_train), color="blue")
plt.title("Salary VS Exprience")
plt.xlabel('years of Exp')
plt.ylabel('salary')
plt.show()


bais = regresor.score(x_train,y_train)
print(bais)

variance = regresor.score(x_test,y_test)
print(variance)

dataset.mean()
dataset['Salary'].mean()

dataset.median()
dataset['Salary'].median()

dataset['Salary'].mode()

dataset.var()
dataset['Salary'].var()

dataset.std()
dataset['Salary'].std()

# coeficient variation (cv)
# for calculating cv we import laibary first 
from scipy.stats  import variation

variation(dataset.values)

variation(dataset["Salary"])

# corelation 
dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])

# skewness
dataset.skew() # this will give us skewness of the particular
dataset['Salary'].skew()

# standard error
dataset.sem() # this will give standard error from entire dataset 
dataset['Salary'].sem()  # this will give us standard error from particular colanm

# Z-score
import scipy.stats as stats
dataset.apply(stats.zscore)

# Degree of freedom

a = dataset.shape[0]  # this will give us no. of rows 
a
b = dataset.shape[1] # this will give no. of column
b
Degree_of_freedom = a-b
print(Degree_of_freedom)


y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

r_squre = 1-(SSR/SST)
r_squre

print(regresor)

bias = regresor.score(x_train,y_train)
print (bias)

variance = regresor.score(x_test,y_test)
print(variance)

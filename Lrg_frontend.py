import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle


#load dataset
dataset = pd.read_csv(r"C:\Users\DELL\Downloads\Salary_Data.csv")

#  split dataset into x and y

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

regresor = LinearRegression()
regresor.fit(x_train,y_train)

y_pred=regresor.predict(x_test)

comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison) 

# visualize the training  set
plt.scatter(x_train,y_train, color="red")
plt.plot(x_train,regresor.predict(x_train), color="blue")
plt.title("Salary VS Exprience (Training set)")
plt.xlabel('years of Exp')
plt.ylabel('salary')
plt.show()

# visualize the test set 
plt.scatter(x_test,y_test, color="red")
plt.plot(x_train,regresor.predict(x_train), color="blue")
plt.title("Salary VS Exprience (Test set)")
plt.xlabel('years of Exp')
plt.ylabel('salary')
plt.show()

y_12 = regresor.predict([[12]])
y_20 = regresor.predict([[20]])
print(f"predict salary of 12 years Exp: ${y_12[0]:,.2f}")
print(f"predict salary of 20 years Exp: ${y_20[0]:,.2f}")

# check model perfomence

bais = regresor.score(x_train,y_train)
variance = regresor.score(x_test,y_test)
train_mse = mean_squared_error(y_train,regresor.predict(x_train))
test_mse = mean_squared_error(y_test,y_pred)

print(f"Training Score (R^2): {bais:.2f}")
print(f"Training Score (R^2): {variance:.2f}")
print(f"Train mse: {train_mse:.2f}")
print(f"test mse: {test_mse:.2f}")

#save the train model the disk 

import pickle
filename = 'Linear_reressor_model.pkl'
with open (filename,'wb')as file:
    pickle.dump(regresor,file)
print ("model has been pickled and saved as linear_regression_model.pkl")

import os
print(os.getcwd())


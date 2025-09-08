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

m_slope = regresor.coef_
print(m_slope)

c_intercept =regresor.intercept_
print(c_intercept ) 

y_12 = m_slope*12 + c_intercept
print(y_12)
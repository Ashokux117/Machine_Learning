#----Data preprocessing pipeline------#

# load libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# load dataset 

dataset = pd.read_csv(r"C:\Users\DELL\Downloads\Dataa.csv")
dataset
# split data into x and y
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#  transformer to  fill missing value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()

imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:,1:3])

# lebel data into categorical to numerical   using lebel encoder
from sklearn.preprocessing import LabelEncoder

le_x = LabelEncoder()
le_x.fit_transform(x[:,0])
x[:,0]= le_x.fit_transform(x[:,0])

# transfer using for dependent variable
le_y = LabelEncoder()
y = le_y.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)









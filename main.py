import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


data = pd.read_csv('tourism_data_500_points.csv')

data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(lambda x: x.strftime("%Y/%m/%d, %H:%M:%S")) 

data[['date','Time']] = data['Date'].str.split(',',expand=True)

data = data.drop('Date', axis = 1)

data.loc[data['Number_of_Visitors'] <= 0]

data['Number_of_Visitors'] = data['Number_of_Visitors'].replace(-166, 166)

data = pd.get_dummies(data)

X = data.drop('Number_of_Visitors', axis = 1)
y = data['Number_of_Visitors']

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model_lasso = Lasso()

model_lasso.fit(x_train,y_train)

lasso_predict =  model_lasso.predict(x_test)


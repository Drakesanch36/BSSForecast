from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np


import warnings
warnings.filterwarnings("ignore")

# import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

#import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split
# from feature_selector import FeatureSelector

#initiate plotly
# pyoff.init_notebook_mode()


data = pd.read_csv("UAH_customer_order_data.csv")
desc = data.describe()
# print(data.head(10))
data['Paid at'] = pd.to_datetime(data['Paid at'])
# data['Year'] = data['Paid at'].dt.year
# data['Month'] = data['Paid at'].dt.month
# data['Day'] = data['Paid at'].dt.day
# data['dayofyear'] = data['Paid at'].dt.dayofyear
# data['dayofweek'] = data['Paid at'].dt.dayofweek
# data['weekofyear'] = data['Paid at'].dt.weekofyear

df_sales = data[['Paid at','Lineitem quantity','Lineitem price']].dropna()
print(df_sales)
df_sales['Sales'] = df_sales['Lineitem quantity'] * df_sales['Lineitem price']
# print(df_sales.head(10))

#represent month in date field as its first day
df_sales['Paid at'] = df_sales['Paid at'].dt.year.astype('str') + '-' + df_sales['Paid at'].dt.month.astype('str') + '-01'
df_sales['Paid at'] = pd.to_datetime(df_sales['Paid at'])
#groupby date and sum the sales
df_sales = df_sales.groupby('Paid at')['Lineitem quantity'].sum().reset_index()
print(len(df_sales))

#plot monthly sales
# plot_data = [
#     go.Scatter(
#         x=df_sales['Paid at'],
#         y=df_sales['Lineitem quantity'],
#     )
# ]
# plot_layout = go.Layout(
#         title='Montly Sales'
#     )
# fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.iplot(fig)

#create a new dataframe to model the difference
df_diff = df_sales.copy()
#add previous sales to the next row
df_diff['prev_sales'] = df_diff['Lineitem quantity'].shift(1)
#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['Lineitem quantity'] - df_diff['prev_sales'])
df_diff.head(10)

#plot sales diff
# plot_data = [
#     go.Scatter(
#         x=df_diff['Paid at'],
#         y=df_diff['diff'],
#     )
# ]
# plot_layout = go.Layout(
#         title='Montly Sales Diff'
#     )
# fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.iplot(fig)

#create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_sales'],axis=1)
#adding lags
for inc in range(1,5):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
#drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)

# print(df_supervised.head(10))

# Import statsmodels.formula.api
import statsmodels.formula.api as smf
# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4', data=df_supervised)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)

#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['Lineitem quantity','Paid at'],axis=1)
#split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False)

y_pred = model.predict(X_test,batch_size=1)
#for multistep prediction, you need to replace X_test values with the predictions coming from t-1

#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print(np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_sales[-7:]['Paid at'])
act_sales = list(df_sales[-7:]['Lineitem quantity'])
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)
#for multistep prediction, replace act_sales with the predicted sales

print(df_result)

#merge with actual sales dataframe
df_sales_pred = pd.merge(df_sales,df_result,on='Sales',how='left')
#plot actual and predicted
plot_data = [
    go.Scatter(
        x=df_sales_pred['Paid at'],
        y=df_sales_pred['Lineitem quantity'],
        name='actual'
    ),
        go.Scatter(
        x=df_sales_pred['Paid at'],
        y=df_sales_pred['pred_value'],
        name='predicted'
    )
    
]
plot_layout = go.Layout(
        title='Sales Prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
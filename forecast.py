import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
# from feature_selector import FeatureSelector
#this UAH Customer order data includes the hour
data = pd.read_csv("UAH_customer_order_data2.csv")
desc = data.describe()
# print(desc)
data['Paid at'] = pd.to_datetime(data['Paid at'])
data['Year'] = data['Paid at'].dt.year
data['Month'] = data['Paid at'].dt.month
data['Day'] = data['Paid at'].dt.day
data['dayofyear'] = data['Paid at'].dt.dayofyear
data['dayofweek'] = data['Paid at'].dt.dayofweek
data['weekofyear'] = data['Paid at'].dt.weekofyear
data['Hour'] = data['Paid at'].dt.hour


fcd = data[['Lineitem price','Lineitem quantity','Year','Month','Day', 'dayofyear','dayofweek','weekofyear','Hour']].dropna()
fcd['Sales'] = fcd['Lineitem price'] * fcd['Lineitem quantity']
#print(fcd)
fcd2 = fcd.groupby(['Month'])['Sales'].sum().reset_index()
print('Monthly Sales:')
print(fcd2)
fcd3 = fcd.groupby(['Hour'])['Lineitem quantity'].count()
print('Orders by the hour:')
print(fcd3)
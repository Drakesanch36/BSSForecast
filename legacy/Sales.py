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


fcd = data[['Name','Local','Lineitem sku','Lineitem price','Lineitem quantity','Year','Month','Day', 'dayofyear','dayofweek','weekofyear','Hour']].dropna()
fcd['Sales'] = fcd['Lineitem price'] * fcd['Lineitem quantity']
#print(fcd)
# fcd2 = fcd.groupby(['Month'])['Sales'].sum().reset_index()
fcd2 = fcd.groupby(['Month']).agg({'Sales':'sum'}).reset_index()
print('Monthly Sales:')
print(fcd2)
fcd2.to_excel (r'C:\Users\drake\Documents\My Tableau Repository\MonthlySales.xlsx', index = False, header=True)
# fcd3 = fcd.groupby(['Hour'])['Lineitem quantity'].count().to_frame('Orders').reset_index()
#agg is used to sum the lineitem quanntity with respect to every hour
fcd3 = fcd.groupby(['Hour']).agg({'Lineitem quantity':'sum'}).reset_index()
print('Orders by the hour:')
print(fcd3)
fcd3.to_excel (r'C:\Users\drake\Documents\My Tableau Repository\SalesPerHour.xlsx', index = False, header=True)

# this is to see what combinations of data is used
# new_data = fcd[fcd['Name'].duplicated(keep=False)]
# new_data['Product_Bundle'] = new_data.groupby('Name')['Lineitem sku'].transform(lambda x: ','.join(x))
# new_data1 = new_data[['Name','Product_Bundle']].drop_duplicates()
# print(new_data1.head())
# from itertools import combinations
# from collections import Counter

# count = Counter()

# for row in new_data1['Product_Bundle']:
#     row_list = row.split(',')
#     count.update(Counter(combination(row_list,2)))
# print(count)
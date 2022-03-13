import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
# from feature_selector import FeatureSelector

data = pd.read_csv("UAH_customer_order_data.csv")
desc = data.describe()
# print(desc)
data['Paid at'] = pd.to_datetime(data['Paid at'])
data['Year'] = data['Paid at'].dt.year
data['Month'] = data['Paid at'].dt.month
data['Day'] = data['Paid at'].dt.day
data['dayofyear'] = data['Paid at'].dt.dayofyear
data['dayofweek'] = data['Paid at'].dt.dayofweek
data['weekofyear'] = data['Paid at'].dt.weekofyear

print(len(data.columns))

fcd = data[['Total','Lineitem name','Lineitem sku','Lineitem quantity','Year','Month','Day', 'dayofyear','dayofweek','weekofyear']].dropna()
# print(fcd)
fcd2 = fcd.groupby(['Lineitem sku','Month'])['Lineitem quantity'].sum().reset_index()
print(fcd2)
x = fcd2.iloc[:, 1:3]
y = fcd2.iloc[:, 0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 0)
import statsmodels.api as sm
ols = sm.OLS(y_train, sm.add_constant(x_train))
lm1 = ols.fit()
print(lm1.summary())
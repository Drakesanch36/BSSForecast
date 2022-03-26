import pandas as pd
from pandas import to_datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from datetime import datetime, date, time, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import prophet
from prophet import Prophet

print(np.__version__)
print(pd.__version__)

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
data['weekofyear'] = data['Paid at'].dt.week
data['Hour'] = data['Paid at'].dt.hour

df = data[['Paid at', 'Lineitem quantity', 'Lineitem price']].dropna().sort_values(by="Paid at").reset_index()
df['y'] = df['Lineitem quantity'] * df['Lineitem price']
df.drop(['Lineitem quantity','Lineitem price'], axis=1)


sf = data[['weekofyear','Lineitem price','Lineitem quantity']].dropna().sort_values(by="weekofyear",ascending=True).reset_index()
sf['Sales'] = sf['Lineitem price'] * sf['Lineitem quantity']
print(df)
# print(sf)

# Average weekly sales

# Overall
avg_weekly_sales = sf.Sales.mean()
print(f"Overall average weekly sales: ${avg_weekly_sales}")

# Last 12 months (this will be the forecasted sales)
avg_weekly_sales_12month = sf.Sales[-12:].mean()
print(f"Last 12 months average weekly sales: ${avg_weekly_sales_12month}")


sf_week = sf.groupby(['weekofyear']).agg({'Sales':'sum'}).reset_index()
# print(sf_week)
# plt.plot(sf_week.weekofyear, sf_week.Sales)
# plt.show()

#end of weekly data

#this will give me the sales week by week
sf1 = data[['Month','Lineitem price','Lineitem quantity']].dropna().sort_values(by="Month",ascending=True)
sf1['Sales'] = sf1['Lineitem price'] * sf1['Lineitem quantity']

# Average weekly sales

# Overall
avg_monthly_sales = sf1.Sales.mean()
print(f"Overall average monthly sales: ${avg_monthly_sales}")

# Last 12 months (this will be the forecasted sales)
avg_monthly_sales_12month = sf1.Sales[-12:].mean()
print(f"Last 12 months average monthly sales: ${avg_monthly_sales_12month}")

sf_month = sf1.groupby(['Month']).agg({'Sales':'sum'}).reset_index()
# print(sf_week)
# plt.plot(sf_month.Month, sf_month.Sales)
# plt.show()
dfcons = df[['Paid at', 'y']].sort_values(by='Paid at',ascending=True)
#columns need to be ds and y for it to run properly
df1 = dfcons.rename(columns={"Paid at": "ds"})
df1['ds'] = to_datetime(df1['ds'])
print(df1)
#defines model
model = Prophet()
#fit the model
model.fit(df1)

# define the period for which we want a prediction
future = list()
for i in range(1, 13):
	date = '2021-%02d' % i
	future.append([date])
future = DataFrame(future)
future.columns = ['ds']
future['ds']= to_datetime(future['ds'])

# summarize the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
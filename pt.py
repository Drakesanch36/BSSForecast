import pandas as pd
from pandas import to_datetime
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
# import seaborn as sns
# %matplotlib inline
# import statsmodels.tsa.api as smt
# import statsmodels.api as sm
# from statsmodels.tools.eval_measures import rmse
from datetime import datetime, date, time, timedelta
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import fbprophet
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

data = pd.read_csv("UAH_customer_order_data.csv")
desc = data.describe()

df = data[['Paid at', 'Lineitem quantity']].dropna().sort_values(by="Paid at").reset_index()
df['y'] = df['Lineitem quantity']
df.drop(['Lineitem quantity'], axis=1)

dfcons = df[['Paid at', 'y']].sort_values(by='Paid at',ascending=True)
#columns need to be ds and y for it to run properly
df1 = dfcons.rename(columns={"Paid at": "ds"})
# df1['ds'] = to_datetime(df1['ds'], format='%Y%m%d')
df2 = df1.groupby(['ds']).agg({'y':'sum'}).reset_index()
# df1.loc[(df1['ds'] > '2021-01-01') & (df1['ds'] < '2022-01-01'), 'y'] = None
print(df2)
#defines model
model = Prophet()
#fit the model
model.fit(df2)

# define the period for which we want a prediction
future = list()
for i in range(1, 13):
	date = '2021-%02d' % i
	future.append([date])
future = DataFrame(future)
future.columns = ['ds']
future['ds']= to_datetime(future['ds'])

#use the model to make a forecast
forecast = model.predict(future)
# summarize the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
model.plot(forecast)
plt.ylim(0,250)
pyplot.show()

# x1 = forecast['ds']
# #x3 = df['ds']
# x2 = df['y']
# y1 = forecast['yhat']
# y2 = forecast['yhat_lower']
# y3 = forecast['yhat_upper']
# #plt.plot(x3,x2)
# fig3 = plt.plot(x1,y3)
# plt.setp(plt.gca(),ylim=(0,400))
# plt.show()

# calculate MAE between expected and predicted values for december
y_true = df['y'][-12:].values
y_pred = forecast['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)
# plot expected vs actual
pyplot.plot(y_true, label='Actual')
pyplot.plot(y_pred, label='Predicted')
pyplot.legend()
pyplot.show()

future1 = model.make_future_dataframe(periods=365)
print(future1.tail)
forecast1 = model.predict(future1)
print(forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

model.plot(forecast1)
plt.show()
model.plot_components(forecast1)
plt.show()

plot_plotly(model, forecast1)
plot_components_plotly(model, forecast1)
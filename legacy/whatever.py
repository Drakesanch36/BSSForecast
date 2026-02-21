import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import plotly as py
from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.offline as offline
import plotly.io as pio


import seaborn as sns
import matplotlib.pyplot as plt

print(np.__version__)
print(pd.__version__)

# from feature_selector import FeatureSelector
#this UAH Customer order data includes the hour
data = pd.read_csv("UAH_customer_order_data.csv")
desc = data.describe()
# print(desc)



ds = data[['Paid at', 'Lineitem price','Lineitem quantity']].dropna()
ds['Sales'] = ds['Lineitem price'] * ds['Lineitem quantity']

alt = ds.drop(['Lineitem quantity', 'Lineitem price'], axis=1)
listing_calendar = alt.rename(columns={'Paid at': 'date', 'Sales': 'price'})
# print(listing_calendar)

listing_calendar.info()
listing_calendar['price'] = pd.to_numeric(listing_calendar['price'], errors = 'coerce')
df_calendar = listing_calendar.groupby('date')[["price"]].sum()
df_calendar['mean'] = listing_calendar.groupby('date')[["price"]].mean()
df_calendar.columns = ['Total', 'Avg']
print(df_calendar.head(10))

#SET DATE AS INDEX
df_calendar2 = listing_calendar.set_index("date")
df_calendar2.index = pd.to_datetime(df_calendar2.index)
df_calendar2 = df_calendar2[['price']].resample('M').mean().reset_index()
print(df_calendar2.head())

trace3 = go.Scatter(
    x = df_calendar2.index[:-1],
    y = df_calendar2.price[:-1]
)
layout3 = go.Layout(
    title = "Average Prices by Month",
    xaxis = dict(title = 'Month'),
    yaxis = dict(title = 'Price ($)')
)
data3 = [trace3]
figure3 = go.Figure(data = data3, layout = layout3)
offline.iplot(figure3)
plt.show()
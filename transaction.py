import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("UAH_customer_order_data.csv")
desc = data.describe()
car = data[['Company','Paid at','Lineitem quantity','Lineitem sku']].dropna()
car['Paid at'] = pd.to_datetime(car['Paid at'])
car['Year'] = car['Paid at'].dt.year
car['Month'] = car['Paid at'].dt.month
car['Day'] = car['Paid at'].dt.day
car1 = car.sort_values(by = 'Paid at', axis = 0, ascending = False, ignore_index = True)
car2 = car1.groupby(['Year','Month','Lineitem sku'])['Company'].value_counts().to_frame('Count').reset_index()
pd.set_option('max_rows', None)
# print(car2)
car3 = car2.groupby(['Lineitem sku','Company'])['Count'].sum().to_frame('Transactions').reset_index()
car3['Transaction Rate'] = car3.loc[:,'Transactions':].sum(axis=1)/12
car4 = car3.sort_values(by=['Company','Transaction Rate'], axis=0, ascending = False, ignore_index=True)
print(car4)


qty = car1.groupby(['Year','Month','Lineitem sku','Company'])['Lineitem quantity'].sum().to_frame('Order QTY').reset_index()
# qty['Monthly Units Rate'] = qty.loc[:,'Order QTY':].sum(axis=1)/12
# qty2 = qty.sort_values(by=['Company','Monthly Units Rate'], axis=0, ascending = False, ignore_index=True)
qty2 = qty.groupby(['Company','Lineitem sku'])['Order QTY'].sum().to_frame('Total Order QTY').reset_index()
qty2['Monthly Units Rate'] = qty2.loc[:,'Total Order QTY':].sum(axis=1)/12
qty3 = qty2.sort_values(by=['Company','Monthly Units Rate'], axis=0, ascending = False, ignore_index=True)
print(qty3)
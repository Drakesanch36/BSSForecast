import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("UAH_customer_order_data.csv")
datapd = pd.read_csv("UAH_Product_Data.csv", encoding= 'unicode_escape')
desc = data.describe()

data.rename(columns = {'Lineitem sku':'sku'}, inplace = True)
data['Paid at'] = pd.to_datetime(data['Paid at'])
data['Year'] = data['Paid at'].dt.year
data['Month'] = data['Paid at'].dt.month
data['Day'] = data['Paid at'].dt.day
data['dayofyear'] = data['Paid at'].dt.dayofyear
data['dayofweek'] = data['Paid at'].dt.dayofweek
data['weekofyear'] = data['Paid at'].dt.weekofyear
data['Hour'] = data['Paid at'].dt.hour


cust = data[['Customer Type','Lineitem quantity','Lineitem price','Month','sku']].dropna()
#print(cpp_data)
#cpp_data.info()
cust['Sales'] = cust['Lineitem quantity'] * cust['Lineitem price']
clt = cust.groupby(['Customer Type'], as_index=False)['Sales'].sum()
clt_order = clt.sort_values(by='Sales', ascending = False)
tt_custom = np.sum(clt_order.loc[:,'Sales':].values)
clt_order['% Total Sales'] = clt_order.loc[:,'Sales':].sum(axis=1)/tt_custom*100
            #displays all data
pd.set_option("display.max_rows", None, "display.max_columns", None)
print('Top 10 Customers:')
print(clt_order.head(n=25).to_string(index=False))
# clt_order.to_excel (r'C:\Users\drake\Documents\My Tableau Repository\top10customers.xlsx', index = False, header=True)

datapd2 = datapd[['sku','category_name']].dropna()
combine = pd.merge(datapd2, cust, how="left", on="sku").dropna()
sku_cons = combine[['sku','Month','Customer Type','Lineitem quantity','category_name']]

fcd2 = sku_cons.groupby(['Month','Customer Type','category_name']).agg({'Lineitem quantity':'count'})
print('Monthly Sales:')
print(fcd2)
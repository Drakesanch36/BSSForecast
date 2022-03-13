import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("UAH_customer_order_data.csv")
desc = data.describe()
# print(data)
#company purchasing data is cpp
cpp_data = data[['Company','Total']].dropna()
#print(cpp_data)
#cpp_data.info()
clt = cpp_data.groupby(['Company'], as_index=False)['Total'].sum()
clt_order = clt.sort_values(by='Total', ascending = False)
tt_custom = np.sum(clt_order.loc[:,'Total':].values)
clt_order['% Total Sales'] = clt_order.loc[:,'Total':].sum(axis=1)/tt_custom*100
            #displays all data
# pd.set_option("display.max_rows", None, "display.max_columns", None)
print('Top 10 Customers:')
print(clt_order.head(n=10).to_string(index=False))


#this displays the most popular bought products
sku_data2 = data[['Total','Lineitem name','Lineitem sku']].dropna()
sku_data2.rename(columns = {'Lineitem sku':'sku'}, inplace = True)
sku = sku_data2.groupby(['Lineitem name','sku'], as_index=False)['Total'].sum()
sku_order = sku.sort_values(by='Total', ascending = False)
tt_sku = np.sum(sku_order.loc[:,'Total':].values)
sku_order['% Total Sales'] = sku_order.loc[:,'Total':].sum(axis=1)/tt_sku*100
            #displays all data
# pd.set_option("display.max_rows", None, "display.max_columns", None)
print('Highest Selling Prodocts:')
print(sku_order.head(n=10).to_string(index=False))


#shows the most popular sku for each company
cpp_data2 = data[['Company','Total','Lineitem name','Lineitem sku']].dropna()
clt2 = cpp_data2.groupby(['Company','Lineitem sku'], as_index=False)['Total'].sum()
clt_order2 = clt2.sort_values(by='Total', ascending = False)
clt_duplicate = clt_order2.drop_duplicates(subset=['Company'], keep='first')
            #displays all data
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print('Top SKU for each Company:')
print(clt_duplicate.head(n=10).to_string(index=False))

        #this data shows me the top ten most searched products
data2 = pd.read_csv("UAH_Search.csv")
desc2 = data2.describe()
#print(data2)
ds2 = data2.sort_values(by='Search count', ascending = False)
print("Top 10 Most Searched Products:")
print(ds2.head(n=10).to_string(index=False))


        #this data gives a list of product names and sku values for each product
# data3 = pd.read_csv("UAH_Product_Data.csv", encoding= 'unicode_escape')
# desc2 = data3.describe()
# print(data3)
# data3.info()

cpp_data2 = data[['Company','Total','Lineitem name','Lineitem sku']].dropna()
datapm = pd.read_csv("UAH_Category_PM.csv", encoding= 'unicode_escape')
datapd = pd.read_csv("UAH_Product_Data.csv", encoding= 'unicode_escape')
datapd2 = datapd[['sku','category_name']].dropna()
datapd2['profit margin'] = datapd2['category_name'].map(datapm.set_index('Category')['Profit Margin'])

sku_df = pd.merge(datapd2, sku, how="left", on="sku").dropna()
sku_cons = sku_df[['sku','category_name','profit margin','Total']]
sku_cons['Profit'] = sku_cons['Total'] * sku_cons['profit margin']
dfStyler = sku_cons.style.set_properties(**{'text-align': 'left'})
dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
sku_cons1 = sku_cons.groupby(['category_name','profit margin'], as_index=False)['Profit'].sum()
sku_cons2 = sku_cons1.sort_values(by='Profit', axis = 0, ascending = False, ignore_index = True)
print("Top 10 Most Profitable Products:")
print(sku_cons2.head(n=10))
# #print (datapd2.head)
# datapd2['Profits'] = datapd2['sku'].map(sku.set_index('Lineitem sku')['Total'])
# print(datapd2)

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


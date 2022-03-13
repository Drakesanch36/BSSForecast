import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("UAH_customer_order_data.csv")
desc = data.describe()
cpp_data2 = data[['Company','Total','Lineitem name','Lineitem sku']].dropna()
clt2 = cpp_data2.groupby(['Company','Lineitem sku'])['Total'].sum().to_frame('Total').reset_index()
clt_order2 = clt2.sort_values(by =['Company','Total'], axis = 0, ascending = False, ignore_index = True)
# clt_duplicate = clt_order2.drop_duplicates(subset=['Company'], keep='first')
            #displays all data
pd.set_option("display.max_rows", None, "display.max_columns", None)
print('Top SKU for each Company:')
print(clt_order2)
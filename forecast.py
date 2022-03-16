import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta,date
from __ future__ import division
import warnings
warnings.filterwarnings("ignore")
import plotyly.plotyly as py
import plotly.offline as pyoff
import plotly.graph_objects as go
import plotly.express as px

import keras
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
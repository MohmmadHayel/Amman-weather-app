# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 09:43:48 2026

@author: User
"""

import pandas as pd
data1=pd.read_csv('scrabed_data.csv')
data2=pd.read_csv('scrabed_data_test.csv')

CombinedData=pd.concat([data1,data2])
describe=CombinedData.describe(include='all')

print(CombinedData['time'].value_counts())

# handling time column
CombinedData['time1'] = CombinedData['time'].str[0:5]
print(CombinedData['time1'].isna().sum())

describe=CombinedData.describe(include='all')
print(CombinedData['time1'].head())
print(CombinedData['time1'].unique())

## handling thetempreture column


print(CombinedData['temperature'].head())
CopyData=CombinedData.copy()
CopyData["temperature"] = pd.to_numeric(CopyData["temperature"].str.extract('(\d+)')[0], errors='coerce').astype(float)
print(CopyData['temperature'].head())
print(CopyData['temperature'].isna().sum())
print(CombinedData['temperature'].isna().sum())
# print(CopyData['temperature'].info())
CopyData["Humidity2"] = pd.to_numeric(CopyData["Humidity"].str.extract('(\d+)')[0], errors='coerce').astype(float)
print(CopyData['Humidity2'].head())
print(CombinedData['Humidity'].isna().sum())


## handling thetempreture column
print(CopyData.columns)

CopyData["wind speed2"] = CopyData["wind speed"].str.replace("km/h", "")
# print(CombinedData["wind speed"].isna().sum())

# CopyData["wind speed2"] = CopyData["wind speed2"].str.replace("No wind", 0)
print(CopyData["wind speed2"].dtypes)


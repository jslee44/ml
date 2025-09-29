import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,date,time

path = './대전태평(암반)/'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.xlsx')]
file_list_py

df = pd.DataFrame()
for i in file_list_py:
  data = pd.read_excel(path + i)
  df = pd.concat([df,data])

df = df.reset_index(drop=True)

df['Date'] = pd.to_datetime(df['날짜'],format='%Y%m%d') + \
             pd.to_timedelta(df['시간'].astype(int),unit='h')

df.set_index(df['Date'],inplace=True)

df1 = df[['수온(℃)','수위(el.m)','EC(㎲/㎝)']]
df1.columns = ['temp','level','EC']

df1.isnull().sum()

df1.to_csv('대전지하수.csv', encoding='cp949')
df = pd.read_csv('대전지하수.csv',index_col='Date',parse_dates=True,encoding='cp949')
df.describe()

df.hist(bins=50,figsize=(10,6))
plt.show()

df.boxplot()

plt.subplot(1,3,1)
df.boxplot(column='temp',return_type='both')
plt.subplot(1,3,2)
df.boxplot(column='level',return_type='both')
plt.subplot(1,3,3)
df.boxplot(column='EC',return_type='both')
plt.show()

import matplotlib.pyplot as plt

# 전체 그림 크기 지정
plt.figure(figsize=(10, 8))
df = df.sort_index()

plt.subplot(3, 1, 1)
df['temp'].plot()
plt.subplot(3, 1, 2)
df['level'].plot()
plt.subplot(3, 1, 3)
df['EC'].plot()

# subplot 간격 자동 조절
plt.tight_layout()
plt.show()
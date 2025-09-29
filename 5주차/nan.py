import pandas as pd
import numpy as np

df = pd.DataFrame({'A':[1,2,np.nan,4,5],
                   'B':[6,7,8,np.nan,10],
                   'C':[11,12,13,np.nan,np.nan]})
pd.isna(df)
pd.isna(df).sum()

df_drop_nan = df.dropna()
df_drop_nan

df_drop_B_C = df.dropna(subset=['B','C'])
df_drop_B_C

df_0 = df['C'].fillna(0)
print(df_0)

df_missing = df['A'].fillna('missing')
df_missing

df_mean = df.fillna(df.mean( ))
print(df,'\n')
print(df_mean)

print(df,'\n')
#결측치 바로 위의 값으로 대체하기
df_ffill = df.fillna(method='ffill')
print(df_ffill,'\n')
#결측치 바로 아래의 값으로 대체하기
df_bfill = df.bfill()
print(df_bfill)

fill_dict = {'A':df['A'].mean(), 'B':'12/25','C':'missing'}
df_filled = df.fillna(value=fill_dict)
df_filled
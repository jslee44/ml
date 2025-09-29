import pandas as pd
import numpy as np

df = pd.read_csv('auto-mpg.csv',header=None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

df['horsepower']=df['horsepower'].replace('?',np.nan)
df = df.dropna(subset=['horsepower'],axis=0)
df['horsepower']=df['horsepower'].astype('float')

df['horsepower'].describe()

df['horsepower_minmax']=(df['horsepower'] - df['horsepower'].min()) / \
                        (df['horsepower'].max() - df['horsepower'].min())
df['horsepower_minmax'].head()
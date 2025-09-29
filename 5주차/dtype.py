import pandas as pd

df = pd.read_csv('auto-mpg.csv',header=None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

df['horsepower']=df['horsepower'].replace('?',np.nan)
df = df.dropna(subset=['horsepower'],axis=0)
df['horsepower']=df['horsepower'].astype('float')
print(df['horsepower'].dtypes)
print(df['horsepower'].unique())

print(df['origin'].unique())

df['origin']=df['origin'].replace({1:'USA',2:'EU',3:'JAPAN'})
print(df['origin'].unique())
print(df['origin'].dtypes)
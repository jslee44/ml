import pandas as pd

df = pd.read_csv('auto-mpg.csv',header=None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']
mpg_to_kpl = 1.60934/3.78541
df['kpl']=df['mpg']*mpg_to_kpl
df.head(3)
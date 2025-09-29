import pandas as pd

df2 = pd.DataFrame({'c1':['a','a','b','a','b'],
                    'c2':[1,1,1,2,2],
                    'c3':[1,1,2,2,2]})
print(df2)
df4 = df2.drop_duplicates(keep=False)
df4
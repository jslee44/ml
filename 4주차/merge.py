import pandas as pd

df3 = pd.DataFrame({'name':['Jessica','Liam','Sophia','Ryan','Alex'],
                    'gender':['F','M','F','M','F'],
                    'age':[21,23,20,22,20]})
df4 = pd.DataFrame({'name':['Sophia','Liam','Ryan','ALex','Jessica'],
                    'department':['CS','MATH','LAW','CS','LAW']})
display(df3,df4)
df5 = pd.merge(df3,df4)
display(df5)
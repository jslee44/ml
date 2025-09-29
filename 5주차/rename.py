import pandas as pd

df = pd.DataFrame({'name':['Jessica','Liam','Sophia','Ryan','Alex'],
                   'test':[45,30,40,37,48],
                   'assign1':[20,17,22,18,24],
                   'assign2':[19,14,18,15,25]})

df['sum'] = df['test']+df['assign1']+df['assign2']
df.rename(columns={'test':'exam'}, inplace=True)
print(df)
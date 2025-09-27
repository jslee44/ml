import pandas as pd

df = pd.DataFrame({'test':[45,30,40,37,48],
                   'assign1':[20,17,22,18,24],
                   'assign2':[19,14,18,15,25]})
display(df)
print(df.describe())
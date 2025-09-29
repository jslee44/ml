import numpy as np

df['grade'] = np.where(df['sum']>=90,'A',
              np.where(df['sum']>=80,'B','C'))
df['result'] = np.where(df['sum']>=80, 'pass','fail')
print(df['result'].value_counts( ))
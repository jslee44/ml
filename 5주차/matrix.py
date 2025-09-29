import seaborn as sns

df = sns.load_dataset('titanic')
df.info()

import missingno as msno
import matplotlib.pyplot as plt

msno.matrix(df)
plt.show()
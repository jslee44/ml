import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')

msno.dendrogram(df)
plt.show()
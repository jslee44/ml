import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

table = titanic.pivot_table(
    index='sex',
    columns='class',
    aggfunc='size',
    observed=False 
)
sns.heatmap(table, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
plt.show()
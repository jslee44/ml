import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')
sns.jointplot(x='age', y='fare', data=titanic)
plt.show()
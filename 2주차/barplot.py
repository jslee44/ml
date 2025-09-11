import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')
sns.barplot(x='class', y='survived', hue='sex', data=titanic)

plt.title('Titanic: Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()
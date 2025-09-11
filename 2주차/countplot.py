import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')
sns.countplot(x='class', hue='sex', data=titanic)

plt.title('Titanic: Passenger Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
sns.relplot(x='total_bill', y='tip', hue='smoker', data=tips)
plt.show()
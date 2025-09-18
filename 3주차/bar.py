import numpy as np
import matplotlib.pyplot as plt

nation = ['Korea', 'Australia', 'Japan', 'Netherlands']
men = [176, 179, 172, 184]
women = [163, 165, 159, 170]
ind = np.arange(len(nation))

plt.figure(figsize=(5,3))
plt.bar(ind, men, color='b', label='men')
plt.bar(ind, women, color='r', label='women')
plt.xticks(ind, nation)
plt.ylim(130, 190)
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

dice = np.random.randint(1, 7, size=1000)
counts = np.bincount(dice)[1:]  # 0은 제외

plt.bar(np.arange(1, 7), counts)
plt.title('Distribution of Dice Rolls (1000 Rolls)')
plt.xlabel('Dice Face')
plt.ylabel('Frequency')
plt.show()
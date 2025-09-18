import numpy as np
import matplotlib.pyplot as plt

year = [2017, 2018, 2019, 2020, 2021]
spring = np.array([124.9, 383.5, 175.0, 173.7, 330.5])
summer = np.array([612.7, 620.6, 508.2, 1037.6, 612.8])
autumn = np.array([177.9, 351.3, 440.8, 270.4, 256.4])
winter = np.array([75.2, 68.7, 168.8, 47.8, 13.3])

plt.bar(year, spring, label='spring')
plt.bar(year, summer, label='summer')
plt.bar(year, autumn, label='autumn')
plt.bar(year, winter, label='winter')
plt.legend()
plt.show()
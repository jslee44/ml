import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(1000)

plt.figure(figsize=(5,3))
plt.hist(x,bins=40, alpha=0.5)
plt.ylabel('frequency')
plt.show()
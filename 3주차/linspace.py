import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1,100)
y = x**2

plt.figure(figsize=(5,3))
plt.plot(x, y, label='y=x^2')
plt.legend()
plt.show()
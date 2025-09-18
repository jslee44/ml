import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.square(np.pi), 100)

plt.subplot(2,1,1)
plt.plot(x, np.sin(x), label='sin')
plt.legend()
plt.subplot(2,1,2)
plt.plot(x, np.cos(x), label='cos')
plt.legend()
plt.show()
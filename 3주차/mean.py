import numpy as np
import matplotlib.pyplot as plt

math = np.array([75,89,75,65,79])
english = np.array([85,80,90,75,88])
korean = np.array([72,90,70,88,93])
x = ['math', 'english', 'korean']

plt.figure(figsize=(5,3))
plt.bar(x, [np.mean(math),np.mean(english),np.mean(korean)])
plt.ylabel('score')
plt.show()
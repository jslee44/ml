import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 11]
y2 = [1, 4, 6, 8, 10]
y3 = [3, 5, 2, 6, 9]

plt.plot(x, y1, marker='o', label='Series A')
plt.plot(x, y2, marker='*', label='Series B')
plt.plot(x, y3, marker='^', label='Series C')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
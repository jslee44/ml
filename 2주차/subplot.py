import matplotlib.pyplot as plt

x=[1,2,3,4,5,6,7,8,9]
y1 = [i * 5 for i in x]
y2 = [i * 1 for i in x]
y3 = [i * 0.3 for i in x]
y4 = [i * 0.2 for i in x]

plt.subplot(2,2,1)
plt.plot(x,y1)

plt.subplot(2,2,2)
plt.plot(x,y2)

plt.subplot(2,2,3)
plt.plot(x,y3)

plt.subplot(2,2,4)
plt.plot(x,y4)

plt.show()
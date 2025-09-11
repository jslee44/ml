import matplotlib.pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
gdp = [300, 543, 1075, 2862, 5979, 10289, 14958, 18320, 21448]

plt.plot(years, gdp, color='r', marker='*', linestyle='dotted')
plt.title('GDP')
plt.xlabel('year')
plt.ylabel('$')
plt.gca().invert_xaxis()
plt.show()
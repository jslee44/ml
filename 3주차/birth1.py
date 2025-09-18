import numpy as np
import matplotlib.pyplot as plt

year = [2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
boy = [248958,223883,223356,224906,208064,184308,167686,155416,139362,133516]
girl = [235592,212572,212079,213514,198179,173463,159136,147260,132975,127046]

ind = np.arange(len(year))

plt.figure(figsize=(6,3.5))
plt.bar(ind, boy, color='b', label='boy')
plt.bar(ind, girl, bottom=boy, color='r', label='girl')
plt.xticks(ind, year)
plt.legend()
plt.xlabel('year')
plt.ylabel('number of births')
plt.show()
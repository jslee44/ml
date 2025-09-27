import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('기온.csv', encoding='cp949')
df.info()
df.rename(columns={'최저기온(°C)':'min_tem'},inplace=True)
df.rename(columns={'평균기온(°C)':'avg_tem'},inplace=True)
df.rename(columns={'최고기온(°C)':'max_tem'},inplace=True)
df.info()

plt.rc('font',family='NanumGothic')
plt.rcParams['axes.unicode_minus']=False

plt.title('서울특별시 2024년도 기온 변화')
plt.plot(range(1,len(df)+1),df['max_tem'],label='최고기온',c='r')
plt.plot(range(1,len(df)+1),df['avg_tem'],label='평균기온',c='y')
plt.plot(range(1,len(df)+1),df['min_tem'],label='최저기온',c='b')
plt.xlabel('일')
plt.ylabel('기온')
plt.legend()
plt.show()

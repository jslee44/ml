import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('기온.csv', encoding='cp949')
df.info()
df.rename(columns={'최저기온(°C)':'min_tem'},inplace=True)
df.rename(columns={'평균기온(°C)':'avg_tem'},inplace=True)
df.rename(columns={'최고기온(°C)':'max_tem'},inplace=True)
df['일시'] = pd.to_datetime(df['일시'],format='%Y-%m-%d')
df3 = df.set_index('일시')

# 3월 데이터 추출
df_Mar = df3[pd.DatetimeIndex(df3.index).month==3]

# 3월 기온 꺽은선 그래프 생성
plt.title('서울특별시 2024년도 3월 기온 변화기온 변화')
plt.plot(range(1,32),df_Mar['max_tem'],label='최고기온',c='r')
plt.plot(range(1,32),df_Mar['avg_tem'],label='평균기온',c='y')
plt.plot(range(1,32),df_Mar['min_tem'],label='최저기온',c='b')
plt.xlabel('3월')
plt.ylabel('기온')
plt.xlim(1,31)
plt.ylim(-7,22)
plt.legend()
plt.rcParams['figure.figsize']=(100,200)
plt.show()
# 이미지 파일 저장
plt.savefig('서울_3월.png')
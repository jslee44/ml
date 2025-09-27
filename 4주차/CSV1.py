import pandas as pd

list1 = list([['Hur','male','30','183'],
              ['Lee','female','24','162'],
              ['Bae','male','23','179'],
              ['Goo','male','21','182'],
              ['Sea','female','28','160'],
              ['Ram','female','26','163'],
              ['Roo','female','24','157'],
              ['Dae','female','24','172']])
col_names = ['name','sex','age','height']
df = pd.DataFrame(list1, columns = col_names)
print(df)

#디렉토리에 CSV 파일로 저장하기
df.to_csv('./file.csv', header=True, index=False)
# CSV 파일 읽기
df2 = pd.read_csv('./file.csv',sep=',')
print(df2)
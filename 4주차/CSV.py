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
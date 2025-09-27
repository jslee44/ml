import pandas as pd

list1 = list([['Kim','male','20','180'],
              ['Lee','male','21','177'],
              ['Park','female','20','160']])
col_names = ['name','sex','age','height']
a = pd.DataFrame(list1, columns = col_names)
print(a)
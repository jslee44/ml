import numpy as np
import pandas as pd

arr1 = np.array([['Kim','male','20','180'],
                ['Lee','male','21','177'],
                ['Park','female','20','165']])
col_names = ['name','sex','age','height']
a = pd.DataFrame(arr1, columns = col_names)
print(a)
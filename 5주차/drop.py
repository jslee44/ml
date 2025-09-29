import pandas as pd
import numpy as np
# 데이터프레임 생성
df = pd.DataFrame({'name': ['Jessica', 'Liam', 'Sophia', 'Ryan', 'Alex'],
                   'test': [45, 30, 40, 37, 48],
                   'assign1': [20, 17, 22, 18, 24],
                   'assign2': [19, 14, 18, 15, 25]})
# 총합 컬럼 추가
df['sum'] = df['test'] + df['assign1'] + df['assign2']
# grade 컬럼 추가
df['grade'] = np.where(df['sum'] >= 90, 'A',
              np.where(df['sum'] >= 80, 'B', 'C'))
# result 컬럼 추가
df['result'] = np.where(df['sum'] >= 80, 'pass', 'fail')
# result 컬럼 삭제
df.drop(columns='result', inplace=True)
print(df)
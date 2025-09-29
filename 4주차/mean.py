# 넘파이를 사용한 평균
import numpy as np
data = [50, 60, 70, 80, 90]
mean = np.mean(data)

# 판다스를 사용한 평균
import pandas as pd
df = pd.DataFrame({'score': [50, 60, 70, 80, 90]})
mean1 = df['score'].mean()

print(data, mean)
print(df, mean1)
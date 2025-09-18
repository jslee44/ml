import numpy as np

score = np.array([71,80,60,90,65])
result = np.where(score>=90,'A',
                  np.where(score>=80,'B',
                           np.where(score>=70,'C','D')))
print(result)
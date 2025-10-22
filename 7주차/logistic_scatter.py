import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  # Age, EstimatedSalary

# 표준화 전 산점도
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
plt.title('Before Standardization')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

# 표준화 수행
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# 표준화 후 산점도
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='red', alpha=0.5)
plt.title('After Standardization')
plt.xlabel('Age (Standardized)')
plt.ylabel('Estimated Salary (Standardized)')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 데이터 불러오기
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  # Age, Estimated Salary
y = dataset.iloc[:, 4].values       # Purchased

# 표준화
sc = StandardScaler()
X = sc.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 시각화
plt.figure(figsize=(8,6))

# 훈련 데이터 산점도
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Train Set', alpha=0.5)

# 테스트 데이터 산점도
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', label='Test Set', alpha=0.5)

plt.title('Train/Test Split Visualization')
plt.xlabel('Age (Standardized)')
plt.ylabel('Estimated Salary (Standardized)')
plt.legend()
plt.grid(True)
plt.show()
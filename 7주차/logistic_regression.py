import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn

# 데이터를 불러오고, 특성 스케일링을 통해 표준화한다.
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# sklearn의 도구로, 데이터의 평균을 0, 표준편차를 1의 표준 정규분포 형태로 변환
sc = StandardScaler()
# Age와 EstimatedSalary의 스케일 차이(값의 범위 차이)를 제거
X = sc.fit_transform(X)
dataset.head()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)

# scikit-learn 라이브러리를 통해 구현되어 있는 로지스틱 회귀 모델 불러오기
classifier = LogisticRegression()

# fit 함수를 통해 모델을 학습
classifier.fit(X_train,y_train)
w_1 = classifier.coef_
w_0 = classifier.intercept_
print(w_1)
print(w_0)

# predict 함수를 통해 로지스틱 회귀로 어든 예측값을 추출해 테스트 데이터로 평가
y_pred = classifier.predict(X_test)
result = sklearn.metrics.accuracy_score(y_test,y_pred)
print(result)

# 학습 데이터셋을 시각화한다
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

# 예측
Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
Z = Z.reshape(X1.shape)

# 결정 경계 시각화
plt.contourf(X1, X2, Z, alpha=0.3, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# 실제 데이터 포인트 시각화
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# 테스트 데이터셋을 시각화한다
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

# 예측
Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
Z = Z.reshape(X1.shape)

# 결정 경계 시각화
plt.contourf(X1, X2, Z, alpha=0.3, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# 실제 데이터 포인트 시각화
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
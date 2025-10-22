# 기본 라이브러리 불러오기
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# [Step1] 데이터 준비 / 기본 설정
# Breast Cancer 데이터셋 가져오기(출처: UCL ML Repository)
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df = pd.read_csv(uci_path,header=None)
df.head()

# 열 이름 지정
df.columns = ['id','clump','cell_size','cell_shape','adhesion','epithlial',
              'bare_nuclei','chromatin','normal_nucleoli','mitoses','class']

# 데이터셋의 크기
print(df.shape)
df.head()

# [Step2] 데이터 탐색
# 데이터 살펴보기
df.sample(5)

# 데이터 자료형 확인
df.info()

# 데이터 통계 요약 정보 확인
df.describe(include='all')

# 데이터 통계 요약 정보 확인
df.describe(include='all')

# 누락 데이터 확인
df.isnull().sum()

# 중복 데이터 확인
df.duplicated().sum()

# 목표 변수
df['class'].value_counts(normalize=True)

# 목표 변수를 이진 변수로 변환
df['class'] = df['class'].map({2:0,4:1})
df['class'].value_counts(normalize=True)

# hist 시각화
df.hist(figsize=(15,12));

# pairplot 시각화
vis_cols = ['clump','cell_size','cell_shape','chromatin','class']
sns.pairplot(data=df[vis_cols],hue='class');

# [Step3] 데이터 전처리
# 중복 데이터 제거
print('중복 제거 이전:',df.shape)
df = df.drop_duplicates()
print('중복 제거 이후:',df.shape)

# 누락 데이터 제거
print('bare_nuclei 열의 고유값:',df['bare_nuclei'].unique())

df = df.copy()
df['bare_nuclei'] = df['bare_nuclei'].replace('?',np.nan) # '?'을 np.na으로 변경
df = df.dropna(subset=['bare_nuclei'],axis=0)             # 누락 데이터 행을 삭제
df['bare_nuclei'] = df['bare_nuclei'].astype('int')       # 문자열을 숫자형으로 변환

# 데이터 통계 요약정보 확인
df.describe()

# [Step4] 데이터셋 구분
# 속성(변수) 선택
train_features = ['clump','cell_size','cell_shape','adhesion','epithlial',
                 'bare_nuclei','chromatin','normal_nucleoli','mitoses']
X = df[train_features]
y = df['class']

# 설명 변수 데이터를 정규화
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data 와 test data로 구분(7:3 비율)
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)

print('train data 개수:',X_train.shape)
print('test data 개수:',X_test.shape)

# [Step5] Decision Tree 분류 모형 - sklearn 사용
# 모형 객체 생성(criterion='entropy' 적용)
tree_model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)

# train data를 가지고 모형 학습
tree_model.fit(X_train,y_train)

# test data를 가지고 y_hat 예측(분류)
y_hat = tree_model.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])

# 모형 성능 평가 - Confusion Matrix 계산
tree_matrix = metrics.confusion_matrix(y_test,y_hat)
print(tree_matrix)

# Confusion Matrix 시각화
plt.figure(figsize=(8,6))
sns.heatmap(tree_matrix,annot=True,fmt='d',cmap='Greens',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# 모형 성능 평가 - 평가지표 계산
tree_report = metrics.classification_report(y_test,y_hat)
print(tree_report)

# 특성 중요도 출력
features = pd.DataFrame(tree_model.feature_importances_,
                        index=train_features,
                        columns=['Importance'])
features = features.sort_values(by='Importance', ascending=False)
features

# 특성 중요도 시각화
plt.figure(figsize=(10,6))
sns.barplot(x=features.Importance,y=features.index,
            hue=features.index,legend=False,
            palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()












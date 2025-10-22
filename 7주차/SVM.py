# 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')

# 데이터 전처리
# 중복 데이터 제거
df = df.drop_duplicates()

# NaN값이 많은 deck 열을 삭제,
# embarked와 내용이 겹치는 embark_town 열을 삭제
rdf = df.drop(['deck','embark_town'],axis=1)

# age 열에 나이 데이터가 없는 모든 행을 삭제
rdf = rdf.dropna(subset=['age'],how='any',axis=0)

# embarked 열의 NaN 값을 승선도시 중에서 가장 많이 출현한 값으로 치환
most_freq = rdf['embarked'].mode()[0]
rdf['embarked'] = rdf['embarked'].fillna(most_freq)

# 변수 선택
# 분석에 활용할 열(속성) 선택
ndf = rdf[['survived','pclass','sex','age','sibsp','parch','embarked']]

# 원핫인코딩 - 범주형 데이터를 모형이 인식할 수 있도록 숫자형으로 변환
onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf,onehot_sex],axis=1)

onehot_embarked = pd.get_dummies(ndf['embarked'],prefix='town')
ndf = pd.concat([ndf,onehot_embarked],axis=1)
ndf = ndf.drop(['sex','embarked'],axis=1)

# 데이터셋 구분 - 훈련용(train data)/검증용(test data)
# 속성(변수) 선택
X = ndf[['pclass','age','sibsp','parch','female','male',
         'town_C','town_Q','town_S']] # 독립 변수 X
y = ndf['survived']                   # 종속 변숙 Y

# 설명 변수 데이터 정규화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)

print('train data 개수:',X_train.shape)
print('test data 개수:', X_test.shape)

# 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')

# 데이터 전처리
# 중복 데이터 제거
df = df.drop_duplicates()

# NaN값이 많은 deck 열을 삭제,
# embarked와 내용이 겹치는 embark_town 열을 삭제
rdf = df.drop(['deck','embark_town'],axis=1)

# age 열에 나이 데이터가 없는 모든 행을 삭제
rdf = rdf.dropna(subset=['age'],how='any',axis=0)

# embarked 열의 NaN 값을 승선도시 중에서 가장 많이 출현한 값으로 치환
most_freq = rdf['embarked'].mode()[0]
rdf['embarked'] = rdf['embarked'].fillna(most_freq)

# 변수 선택
# 분석에 활용할 열(속성) 선택
ndf = rdf[['survived','pclass','sex','age','sibsp','parch','embarked']]

# 원핫인코딩 - 범주형 데이터를 모형이 인식할 수 있도록 숫자형으로 변환
onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf,onehot_sex],axis=1)

onehot_embarked = pd.get_dummies(ndf['embarked'],prefix='town')
ndf = pd.concat([ndf,onehot_embarked],axis=1)
ndf = ndf.drop(['sex','embarked'],axis=1)

# 데이터셋 구분 - 훈련용(train data)/검증용(test data)
# 속성(변수) 선택
X = ndf[['pclass','age','sibsp','parch','female','male',
         'town_C','town_Q','town_S']] # 독립 변수 X
y = ndf['survived']                   # 종속 변숙 Y

# 설명 변수 데이터 정규화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)

print('train data 개수:',X_train.shape)
print('test data 개수:', X_test.shape)

# 모델 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
svm_matrix = metrics.confusion_matrix(y_test,y_hat)
print(svm_matrix)

# Confusion Matrix 시각화
plt.figure(figsize=(8,6))
sns.heatmap(svm_matrix,annot=True,fmt='d',cmap='OrRd',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# 모형 성능 평가 - 평가지표 계산
svm_report = metrics.classification_report(y_test,y_hat)
print(svm_report)

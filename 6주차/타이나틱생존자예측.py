import numpy as np
import pandas as pd

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
submission = pd.read_csv("./data/gender_submission.csv")

train_df

import random
np.random.seed(1234)
random.seed(1234)

print(train_df.shape)
print(test_df.shape)

train_df.dtypes # 각 칼럼의 데이터타입들

train_df["Sex"].value_counts()

train_df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

train_df[["Embarked", "Survived", "PassengerId"]]

embarked_df = train_df[["Embarked", "Survived", "PassengerId"]].dropna().groupby(["Embarked", "Survived"]).count().unstack()
embarked_df

embarked_df.plot.bar(stacked=True)

# 생존자 수 / 전체 인원 수
embarked_df["survived_rate"] = embarked_df.iloc[:,1] / (embarked_df.iloc[:, 0] + embarked_df.iloc[:,1])
embarked_df["survived_rate"]

sex_df = train_df[["Sex", "Survived", "PassengerId"]].dropna().groupby(["Sex","Survived"]).count().unstack()

sex_df.plot.bar(stacked=True)

ticket_df = train_df[["Pclass", "Survived", "PassengerId"]].dropna().groupby(["Pclass","Survived"]).count().unstack()

ticket_df.plot.bar(stacked=True)

plt.hist(x=[train_df.Age[train_df.Survived==0], train_df.Age[train_df.Survived==1]], bins=8, histtype='barstacked', label=['Death', 'Surveved'])
plt.legend()

train_df_corr = pd.get_dummies(train_df, columns=["Sex"], drop_first=True)
train_df_corr = pd.get_dummies(train_df_corr, columns=["Embarked"])
train_df_corr["Sex_male"] = train_df_corr["Sex_male"].astype(int)
train_df_corr.head()

# 숫자형 데이터만 선택
train_df_numeric = train_df_corr.select_dtypes(include=[np.number])

# 숫자형 데이터의 상관관계 행렬 계산
train_corr = train_df_numeric.corr()

train_corr

plt.figure(figsize=(9,9))
sns.heatmap(train_corr, vmax=1, vmin=-1, center=0, annot=True)

# 데이터 전처리와 특정 값 생성

all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

all_df

all_df.isnull().sum()

Fare_mean = all_df[["Pclass","Fare"]].groupby("Pclass").mean().reset_index()

Fare_mean.columns = ["Pclass", "Fare_mean"]

Fare_mean

all_df = pd.merge(all_df, Fare_mean, on="Pclass", how="left")
all_df.loc[(all_df["Fare"].isnull()),"Fare"] = all_df["Fare_mean"]
all_df = all_df.drop("Fare_mean", axis=1)

all_df.isnull().sum()

# Name 호칭 보기

all_df["Name"].head(5)

name_df = all_df["Name"].str.split("[,.]", expand=True).iloc[:, :3]

name_df.columns  = ["family_name", "honorific", "name"]

name_df

name_df["family_name"]  = name_df["family_name"].str.strip()
name_df["honorific"]  = name_df["honorific"].str.strip()
name_df["name"]  = name_df["name"].str.strip()

name_df["honorific"].value_counts()

all_df = pd.concat([all_df, name_df], axis=1)

all_df

plt.figure(figsize=(18,5))
sns.boxplot(x="honorific", y="Age", data=all_df)







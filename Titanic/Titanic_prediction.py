#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import re
import sklearn.preprocessing as preprocessing
from sklearn import model_selection
from sklearn import linear_model

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
frames = [train, test]
data_all = pd.concat(frames, sort=True, keys=['x', 'y'])


'''
拟合Cabin
'''
# 定义Cabin的拟合方法
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

# 补全Cabin`
set_Cabin_type(data_all)


'''
根据Title分类
'''
# what is each person's title?
data_all['Title'] = data_all['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

# 将各式称呼进行统一化处理：
title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr','Jonkheer'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master'], 'Master'))
data_all['Title'] = data_all['Title'].map(title_Dict)

# #使用dummy对不同的称呼进行分列：
dummies_Title = pd.get_dummies(data_all['Title'], prefix= 'Title')
data_all = pd.concat([data_all, dummies_Title], axis=1)


'''
# 补全Age
'''
master_mean_age = data_all.loc[(data_all.Title_Master == 1), 'Age'].mean()
data_all.loc[(data_all.Age.isnull()) & (data_all.Title_Master == 1), 'Age' ] = master_mean_age

miss_mean_age = data_all.loc[(data_all.Title_Miss == 1), 'Age'].mean()
data_all.loc[(data_all.Age.isnull()) & (data_all.Title_Miss == 1), 'Age' ] = miss_mean_age

mr_mean_age = data_all.loc[(data_all.Title_Mr == 1), 'Age'].mean()
data_all.loc[(data_all.Age.isnull()) & (data_all.Title_Mr == 1), 'Age' ] = mr_mean_age

mrs_mean_age = data_all.loc[(data_all.Title_Mrs == 1), 'Age'].mean()
data_all.loc[(data_all.Age.isnull()) & (data_all.Title_Mrs == 1), 'Age' ] = mrs_mean_age

officer_mean_age = data_all.loc[(data_all.Title_Officer == 1), 'Age'].mean()
data_all.loc[(data_all.Age.isnull()) & (data_all.Title_Officer == 1), 'Age' ] = officer_mean_age

royalty_mean_age = data_all.loc[(data_all.Title_Royalty == 1), 'Age'].mean()
data_all.loc[(data_all.Age.isnull()) & (data_all.Title_Royalty == 1), 'Age' ] = royalty_mean_age


'''
# 添加Child字段
'''
data_all['Child'] = (data_all.Age < 12) + 0


'''
# 完成Age与Fare的归一化
'''
scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(np.array(data_all.Age).reshape(-1,1))
data_all['Age_scaled'] = scaler.fit_transform(np.array(data_all.Age).reshape(-1,1), age_scale_param)

fare_scale_param = scaler.fit(np.array(data_all.Fare).reshape(-1,1))
data_all['Fare_scaled'] = scaler.fit_transform(np.array(data_all.Fare).reshape(-1,1), fare_scale_param)


'''
# 参数类目化
'''
dummies_Cabin = pd.get_dummies(data_all['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_all['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_all['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_all['Pclass'], prefix= 'Pclass')

df = pd.concat([data_all, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1, inplace=True)


'''
# 将总表格分为训练集和测试集
'''
data_train = df.loc[('x',slice(None)),:]
data_test = df.loc[('x',slice(None)),:]
data_test.pop('Survived')


'''
交叉验证集
# 简单看看打分情况
'''
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
data_train = df.filter(regex='Survived|Age_.*|Sibsp|Parch|Fare_scaled|Carbin_*|Sex_*|Pclass_*|Child|Title_*')

X = np.array(data_train.values[:,1:])
y = np.array(data_train.values[:,0]).reshape(-1,1).astype(int)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
y = y.iloc[:,0].map(int)
# k折交叉验证
# 验证某个模型在某个训练集上的稳定性，输出cv=k个预测精度
# 把初始训练样本分成k份，其中（k-1）份被用作训练集，剩下一份被用作评估集，这样一共可以对分类器做k次训练，并且得到k个训练结果。
print(model_selection.cross_val_score(clf, X, y, cv=5))
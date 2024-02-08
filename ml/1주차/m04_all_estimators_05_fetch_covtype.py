from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
#1

datasets = fetch_covtype()

X = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=15152, train_size=0.8)

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgoritms : ', allAlgorithms)
print('모델의 갯수',len(allAlgorithms)) #모델의 갯수 41

for name, algorithm in allAlgorithms:
    
    try:
        #2 모델
        model = algorithm()
        #3 훈련
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 : ', acc)
    except Exception as e:
        print(name , '에러 발생', e)
        continue

# AdaBoostClassifier 의 정답률 :  0.5912842181355042
# BaggingClassifier 의 정답률 :  0.9621782570157397
# BernoulliNB 의 정답률 :  0.6320662977720024
# CalibratedClassifierCV 의 정답률 :  0.6757742915415265
# CategoricalNB 에러 발생 Negative values in data passed to CategoricalNB (input X)
# ClassifierChain 에러 발생 __init__() missing 1 required positional argument: 'base_estimator'
# ComplementNB 에러 발생 Negative values in data passed to ComplementNB (input X)
# DecisionTreeClassifier 의 정답률 :  0.9390549297350327
# DummyClassifier 의 정답률 :  0.489204237412115
# ExtraTreeClassifier 의 정답률 :  0.8642289785977987
# ExtraTreesClassifier 의 정답률 :  0.9541664156691307
# GaussianNB 의 정답률 :  0.45793998433775374
# GaussianProcessClassifier 에러 발생 Unable to allocate 1.57 TiB for an array with shape (464809, 464809) and data type float64
# GradientBoostingClassifier 의 정답률 :  0.7736805418104524
# HistGradientBoostingClassifier 의 정답률 :  0.7813997917437587
# KNeighborsClassifier 의 정답률 :  0.9681763809884426
# LabelPropagation 에러 발생 Unable to allocate 1.57 TiB for an array with shape (464809, 464809) and data type float64
# LabelSpreading 에러 발생 Unable to allocate 1.57 TiB for an array with shape (464809, 464809) and data type float64
# LinearDiscriminantAnalysis 의 정답률 :  0.6812560777260483
# LinearSVC 의 정답률 :  0.5251671643589236
# LogisticRegression 의 정답률 :  0.6229959639596224
# LogisticRegressionCV 의 정답률 :  0.66950939304493
# MLPClassifier 의 정답률 :  0.7329500959527723
# MultiOutputClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimator'
# MultinomialNB 에러 발생 Negative values in data passed to MultinomialNB (input X)
# NearestCentroid 의 정답률 :  0.1935664311592644
# NuSVC 에러 발생 specified nu is infeasible
# OneVsOneClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimator'
# OneVsRestClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimator'
# OutputCodeClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimator'
# PassiveAggressiveClassifier 의 정답률 :  0.39801037838954245
# Perceptron 의 정답률 :  0.4702116124368562
# QuadraticDiscriminantAnalysis 의 정답률 :  0.08326807397399379
# RadiusNeighborsClassifier 에러 발생 No neighbors found for test samples array([     0,      1,      2, ..., 116200, 116201, 116202], dtype=int64), you can try using larger radius, giving a label for outliers, or considering removing them from your dataset.
# RandomForestClassifier 의 정답률 :  0.954768809755342
# RidgeClassifier 의 정답률 :  0.7022968425944253
# RidgeClassifierCV 의 정답률 :  0.702193575036789
# SGDClassifier 의 정답률 :  0.6225140486906534
# SVC 의 정답률 :  0.7156699913083139
# StackingClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimators'
# VotingClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimators'
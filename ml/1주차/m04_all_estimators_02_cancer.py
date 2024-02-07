import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()

X = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=53, train_size=0.8, stratify=y)

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
    
# AdaBoostClassifier 의 정답률 :  0.9824561403508771
# BaggingClassifier 의 정답률 :  0.9649122807017544
# BernoulliNB 의 정답률 :  0.631578947368421
# CalibratedClassifierCV 의 정답률 :  0.9298245614035088
# CategoricalNB 에러 발생 index 39 is out of bounds for axis 1 with size 34
# ClassifierChain 에러 발생 __init__() missing 1 required positional argument: 'base_estimator'
# ComplementNB 의 정답률 :  0.9122807017543859
# DecisionTreeClassifier 의 정답률 :  0.8947368421052632
# DummyClassifier 의 정답률 :  0.631578947368421
# ExtraTreeClassifier 의 정답률 :  0.9122807017543859
# ExtraTreesClassifier 의 정답률 :  0.9736842105263158
# GaussianNB 의 정답률 :  0.9649122807017544
# GaussianProcessClassifier 의 정답률 :  0.8771929824561403
# GradientBoostingClassifier 의 정답률 :  0.9736842105263158
# HistGradientBoostingClassifier 의 정답률 :  1.0
# KNeighborsClassifier 의 정답률 :  0.9122807017543859
# LabelPropagation 의 정답률 :  0.38596491228070173
# LabelSpreading 의 정답률 :  0.38596491228070173
# LinearDiscriminantAnalysis 의 정답률 :  0.9649122807017544
# LinearSVC 의 정답률 :  0.8333333333333334
# LogisticRegression 의 정답률 :  0.956140350877193
# LogisticRegressionCV 의 정답률 :  0.956140350877193
# MLPClassifier 의 정답률 :  0.9385964912280702
# MultiOutputClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimator'
# MultinomialNB 의 정답률 :  0.9122807017543859
# NearestCentroid 의 정답률 :  0.8859649122807017
# NuSVC 의 정답률 :  0.8947368421052632
# OneVsOneClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimator'
# OneVsRestClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimator'
# OutputCodeClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimator'
# PassiveAggressiveClassifier 의 정답률 :  0.9298245614035088
# Perceptron 의 정답률 :  0.9298245614035088
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9736842105263158
# RadiusNeighborsClassifier 에러 발생 No neighbors found for test samples array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
#         13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
#         26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
#         39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
#         52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
#         65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
#         78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
#         91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
#        104, 105, 106, 107, 108, 109, 110, 111, 112, 113], dtype=int64), you can try using larger radius, giving a label for outliers, or considering removing them from your dataset.   
# RandomForestClassifier 의 정답률 :  0.956140350877193
# RidgeClassifier 의 정답률 :  0.9736842105263158
# RidgeClassifierCV 의 정답률 :  0.9649122807017544
# SGDClassifier 의 정답률 :  0.6228070175438597
# SVC 의 정답률 :  0.9298245614035088
# StackingClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimators'
# VotingClassifier 에러 발생 __init__() missing 1 required positional argument: 'estimators
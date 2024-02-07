# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import random
import time
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

X = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
test_csv = test_csv.drop([], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=15)
#2. 모델
allAlgorithms = all_estimators(type_filter='regressor')
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
    
# ARDRegression 의 정답률 :  0.2610488889492787
# AdaBoostRegressor 의 정답률 :  0.21478193290275271
# BaggingRegressor 의 정답률 :  0.24104581969545213
# BayesianRidge 의 정답률 :  0.261262505008622
# CCA 에러 발생 n_components == 2, must be <= 1.
# DecisionTreeRegressor 의 정답률 :  -0.13638456154277856
# DummyRegressor 의 정답률 :  -9.202194678659126e-05
# ElasticNet 의 정답률 :  0.26180836108238537
# ElasticNetCV 의 정답률 :  0.25981552125693363
# ExtraTreeRegressor 의 정답률 :  -0.08343354674254999
# ExtraTreesRegressor 의 정답률 :  0.2068618104567317
# GammaRegressor 의 정답률 :  0.18435039978293988
# GaussianProcessRegressor 의 정답률 :  -0.2431734512492687
# GradientBoostingRegressor 의 정답률 :  0.33666367890201854
# HistGradientBoostingRegressor 의 정답률 :  0.3638603605466588
# HuberRegressor 의 정답률 :  0.23909972509728206
# IsotonicRegression 에러 발생 Isotonic regression input X should be a 1d array or 2d array with 1 feature
# KNeighborsRegressor 의 정답률 :  0.21291786763422105
# KernelRidge 의 정답률 :  0.24410914561221642
# Lars 의 정답률 :  0.2608926524303853
# LarsCV 의 정답률 :  0.2613611441191942
# Lasso 의 정답률 :  0.2618734599089063
# LassoCV 의 정답률 :  0.2619421942677024
# LassoLars 의 정답률 :  -9.202194678659126e-05
# LassoLarsCV 의 정답률 :  0.2613611441191942
# LassoLarsIC 의 정답률 :  0.2611817536199392
# LinearRegression 의 정답률 :  0.2608926524303853
# LinearSVR 의 정답률 :  0.22662213415155974
# MLPRegressor 의 정답률 :  0.2851961914090664
# MultiOutputRegressor 에러 발생 __init__() missing 1 required positional argument: 'estimator'
# MultiTaskElasticNet 에러 발생 For mono-task outputs, use ElasticNet
# MultiTaskElasticNetCV 에러 발생 For mono-task outputs, use ElasticNetCVCV
# MultiTaskLasso 에러 발생 For mono-task outputs, use ElasticNet
# MultiTaskLassoCV 에러 발생 For mono-task outputs, use LassoCVCV
# NuSVR 의 정답률 :  0.22118335408110434
# OrthogonalMatchingPursuit 의 정답률 :  0.15687142312956825
# OrthogonalMatchingPursuitCV 의 정답률 :  0.25879428655322156
# PLSCanonical 에러 발생 n_components == 2, must be <= 1.
# PLSRegression 의 정답률 :  0.2619438009968429
# PassiveAggressiveRegressor 의 정답률 :  -0.02955619425310485
# PoissonRegressor 의 정답률 :  0.27224714896459
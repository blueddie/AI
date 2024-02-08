from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1226, train_size=0.9)

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
        # print(name , '에러 발생', e)
        continue
    
# ARDRegression 의 정답률 :  0.7096440734999497
# AdaBoostRegressor 의 정답률 :  0.5982967364247083
# BaggingRegressor 의 정답률 :  0.6034030976166607
# BayesianRidge 의 정답률 :  0.7066575325602795
# CCA 에러 발생 n_components == 2, must be <= 1.
# DecisionTreeRegressor 의 정답률 :  0.27841949004517297
# DummyRegressor 의 정답률 :  -0.010458755922133411
# ElasticNet 의 정답률 :  -0.00044988283613167646
# ElasticNetCV 의 정답률 :  0.5971517438779078
# ExtraTreeRegressor 의 정답률 :  0.18107618151014393
# ExtraTreesRegressor 의 정답률 :  0.6443437313170837
# GammaRegressor 의 정답률 :  -0.0025532594310206935
# GaussianProcessRegressor 의 정답률 :  -8.343464294963512
# GradientBoostingRegressor 의 정답률 :  0.6337019440025938
# HistGradientBoostingRegressor 의 정답률 :  0.5967130071472662
# HuberRegressor 의 정답률 :  0.7114110181353335
# IsotonicRegression 에러 발생 Isotonic regression input X should be a 1d array or 2d array with 1 feature
# KNeighborsRegressor 의 정답률 :  0.6375318525941411
# KernelRidge 의 정답률 :  -2.2069405849162167
# Lars 의 정답률 :  0.7041936103163475
# LarsCV 의 정답률 :  0.705265087972123
# Lasso 의 정답률 :  0.440971317562742
# LassoCV 의 정답률 :  0.7086303905049791
# LassoLars 의 정답률 :  0.4648225559967115
# LassoLarsCV 의 정답률 :  0.7041936103163478
# LassoLarsIC 의 정답률 :  0.7060572818918396
# LinearRegression 의 정답률 :  0.7041936103163478
# LinearSVR 의 정답률 :  -0.13225315686317574
# MLPRegressor 의 정답률 :  -2.1715328606334836
# MultiOutputRegressor 에러 발생 __init__() missing 1 required positional argument: 'estimator'
# MultiTaskElasticNet 에러 발생 For mono-task outputs, use ElasticNet
# MultiTaskElasticNetCV 에러 발생 For mono-task outputs, use ElasticNetCVCV
# MultiTaskLasso 에러 발생 For mono-task outputs, use ElasticNet
# MultiTaskLassoCV 에러 발생 For mono-task outputs, use LassoCVCV
# NuSVR 의 정답률 :  0.21834816617372232
# OrthogonalMatchingPursuit 의 정답률 :  0.4888658834025441
# OrthogonalMatchingPursuitCV 의 정답률 :  0.6925530761190238
# PLSCanonical 에러 발생 n_components == 2, must be <= 1.
# PLSRegression 의 정답률 :  0.6914855310731389
# PassiveAggressiveRegressor 의 정답률 :  0.6707433819277723
# PoissonRegressor 의 정답률 :  0.4214830506541484
# QuantileRegressor 의 정답률 :  -0.0006829453518182316
# RANSACRegressor 의 정답률 :  0.32007453371450123
# RadiusNeighborsRegressor 의 정답률 :  -0.010458755922133411
# RandomForestRegressor 의 정답률 :  0.64123858883451
# RegressorChain 에러 발생 __init__() missing 1 required positional argument: 'base_estimator'
# Ridge 의 정답률 :  0.5730527056915302
# RidgeCV 의 정답률 :  0.6984845876047333
# SGDRegressor 의 정답률 :  0.5611838791796218
# SVR 의 정답률 :  0.2512642784298095
# StackingRegressor 에러 발생 __init__() missing 1 required positional argument: 'estimators'
# TheilSenRegressor 의 정답률 :  0.7274565118456132
# TransformedTargetRegressor 의 정답률 :  0.7041936103163478
# TweedieRegressor 의 정답률 :  -0.0028370834570170533
# VotingRegressor 에러 발생 __init__() missing 1 required positional argument: 'estimators'
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# 1. 데이터
x, y = load_linnerud(return_X_y=True)
# print(x)
# print(y)
# print(x.shape, y.shape) #(20, 3) (20, 3)
# print(np.unique(y, return_counts=True))

# 최종값 ->  x :  [  2. 110.  43.], y =  [138.  33.  68.]

# 2. 모델
# model = RandomForestRegressor()
# model = Ridge()
# model = LinearRegression()
# model = XGBRegressor()
# model = CatBoostRegressor()       # 에러. Currently only multi-regression, multilabel and survival objectives work with multidimensional target
# model = LGBMRegressor()         # 에러. ValueError: y should be a 1d array, got an array of shape (20, 3) instead.

# model = MultiOutputRegressor(LGBMRegressor())
# model = MultiOutputRegressor(CatBoostRegressor())
model = CatBoostRegressor(loss_function='MultiRMSE')

# 3. 훈련
model.fit(x, y)

# 4.
score = model.score(x, y)
print(score)
y_pred = model.predict(x)
print(model.__class__.__name__, "스코어 : ",
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))

# y =  [138.  33.  68.]

# RandomForestRegressor 스코어 :  3.7437
# [[154.67  34.3   63.56]]

# Ridge 스코어 :  7.4569
# [[187.32842123  37.0873515   55.40215097]]

# LinearRegression 스코어 :  7.4567
# [[187.33745435  37.08997099  55.40216714]]

# XGBRegressor 스코어 :  0.0008
# [[138.0005    33.002136  67.99897 ]]

# 0.0
# MultiOutputRegressor 스코어 :  8.91     # LGBM
# [[178.6  35.4  56.1]]

# 0.9994196774909875                     # Catboost
# MultiOutputRegressor 스코어 :  0.2154
# [[138.97756017  33.09066774  67.61547996]]

# 0.999961651757293                       # Catboost loss_function='MultiRMSE'
# CatBoostRegressor 스코어 :  0.0638
# [[138.21649371  32.99740595  67.8741709 ]]
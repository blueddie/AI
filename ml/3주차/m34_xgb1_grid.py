from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import kfold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

### 35분


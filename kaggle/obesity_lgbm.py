# https://www.kaggle.com/competitions/playground-series-s4e2/overview

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm  as lgb
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, RandomizedSearchCV, GridSearchCV
import time
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings

warnings.filterwarnings("ignore")


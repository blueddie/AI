
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support


csv_path = "C:\\_data\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

# train_csv['근로기간'] = train_csv['근로기간'].str.slice(0, 2)
# test_csv['근로기간'] = test_csv['근로기간'].str.slice(0, 2)
# train_csv['근로기간'] = train_csv['근로기간'].str.strip()
# test_csv['근로기간'] = test_csv['근로기간'].str.strip()

print(pd.value_counts(train_csv['대출등급']))
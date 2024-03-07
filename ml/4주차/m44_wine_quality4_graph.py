import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

path  = 'C:\\_data\\dacon\\wine\\'

# 1. 데이터
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(train_csv)

grouped_counts = train_csv.groupby('quality').size()
print(grouped_counts)
# 시각화
plt.bar(grouped_counts.index, grouped_counts.values)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Count of Labels')
plt.show()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#1 . 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)
df['Target(Y)'] = y
print(df)

print("========================== 상관계수 히트맵 ==============================")
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib 
print(sns.__version__)  # 0.12.2
print(matplotlib.__version__)   #tf212: 3.8.0  base : 3.7.2

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()

import sklearn as sk
print(sk.__version__)   #1.1.3

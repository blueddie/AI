import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10]
                    , [2, 4, np.nan, 8, np.nan]
                    , [2, 4, 6, 8, 10]
                    , [np.nan, 4, np.nan, 8, np.nan]
                    ])

# print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 

imputer = SimpleImputer()
data2 = imputer.fit_transform(data) # 평균
print(data2)

imputer = SimpleImputer(strategy='mean')
data3 = imputer.fit_transform(data) # 평균
print(data3)

imputer = SimpleImputer(strategy='median')
data4 = imputer.fit_transform(data) # 중위값
print(data4)

imputer = SimpleImputer(strategy='most_frequent')
data5 = imputer.fit_transform(data) # 가중 자주 나오는 값
print(data5)

imputer = SimpleImputer(strategy='constant')
data6 = imputer.fit_transform(data) # 상수 # 0
print(data6)

imputer = SimpleImputer(strategy='constant', fill_value=777)
data7 = imputer.fit_transform(data) # 상수 # 0
print(data7)

#============================================
print("==============KNNImputer==============")
imputer = KNNImputer()          # KNN 알고리즘
data8 = imputer.fit_transform(data)
print(data8)

imputer = IterativeImputer()        # 선형회귀 알고리즘
data9 = imputer.fit_transform(data)
print(data9)

print(np.__version__)   #   1.26.3 mice 작동 안 함
                        #   1.22.4 mice 작동함

from impyute.imputation.cs import mice

aaa = mice(data.values
           , n=10
           , seed=777
           )
print(aaa)


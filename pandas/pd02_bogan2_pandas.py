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

# 결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

# 결측치 삭제
# print(data.dropna())  # 디폴트 axis=0
# print(data.dropna(axis=0))
# print(data.dropna(axis=1))

# 2-1. 특정 값 - 평균
print("====특정 값 - 평균=====")
means = data.mean()
print(means)
data2 = data.fillna(means)
print(data2)

# 2-2 특정값 - 중위
print("====특정값 - 중위=====")
med = data.median()
print(med)
data3 = data.fillna(med)
print(data3)

# 2-3. 특정값 - 0 채우기
print("====특정값 - 0 채우기=====")
data4 = data.fillna(0)
print(data4)

# 2-4. 특정값 - ffill
print("====특정값 - ffill=====")
data5 = data.ffill()
print(data5)

# 2-5. 특정값 - bfill
print("====특정값 - bfill=====")
data6 = data.bfill()
print(data6)


#   ########### 특정 컬럼만 ###################
means = data['x1'].mean()
print(means)    # 6.5

meds = data['x4'].median()
print(meds)     # 4.0

data['x1'] = data['x1'].fillna(means)
data['x4'] = data['x4'].fillna(meds)
data['x2'] = data['x2'].ffill()
print(data)
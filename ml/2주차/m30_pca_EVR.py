# 스케이릴링, PCA 후 train_test_split
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
print(sk.__version__)   # 1.1.3

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #   (150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=30)   # 
x = pca.fit_transform(x)
print(x)
print(x.shape)

# 통상적으로 PCA 전에는 스탠다드 스케일러를 적용한다. 
x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle=True)

#2 모델
model = RandomForestClassifier(random_state=777)

#3.
model.fit(x_train, y_train)

#4.
results = model.score(x_test, y_test)

print(f"model.score : {results}")

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()
# print(evr)              # 1.0에 가까울 때 제일 좋다
# print(sum(evr)) # 1.0

# (442, 10) (442,)
# model.score : 0.3906543125675058

#============= PCA 후 ===============================
# (442, 10)
# model.score : 0.43734229396217095

# (442, 8)
# model.score : 0.4530900594834356
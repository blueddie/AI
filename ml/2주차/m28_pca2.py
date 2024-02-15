# train_test_split 후 스케일링, PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
print(sk.__version__)   # 1.1.3

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #   (150, 4) (150,)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# pca = PCA(n_components=2)   # 
# x = pca.fit_transform(x)
# print(x)
# print(x.shape)

# 통상적으로 PCA 전에는 스탠다드 스케일러를 적용한다. 
x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle=True, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=2)   
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#2 모델
model = RandomForestClassifier(random_state=777)

#3.
model.fit(x_train, y_train)

#4.
results = model.score(x_test, y_test)
print(x_train.shape)
print(f"model.score : {results}")

# (150, 4)
# model.score : 0.9333333333333333

# (150, 3)
# model.score : 0.9666666666666667

# (150, 2)
# model.score : 0.9333333333333333

# (150, 1)
# model.score : 0.9666666666666667
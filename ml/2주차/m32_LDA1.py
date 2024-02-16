# 스케이릴링, PCA 후 train_test_split
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
print(sk.__version__)   # 1.1.3

# 1. 데이터
# datasets = load_iris()
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #   (150, 4) (150,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

# pca = PCA(n_components=1)   # 
# x = pca.fit_transform(x)

lda = LinearDiscriminantAnalysis(n_components=1)
# ValueError: n_components cannot be larger than min(n_features, n_classes - 1)
x = lda.fit_transform(x, y)
print(x)
print(x.shape)

# 통상적으로 PCA 전에는 스탠다드 스케일러를 적용한다. 
x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle=True, stratify=y)

#2 모델
model = RandomForestClassifier(random_state=777)

#3.
model.fit(x_train, y_train)

#4.
results = model.score(x_test, y_test)

print(f"model.score : {results}")

# (150, 4)
# model.score : 0.9666666666666667

# (150, 3)
# model.score : 0.9333333333333333

# (150, 2)
# model.score : 0.9666666666666667

# (150, 1)
# model.score : 0.9666666666666667

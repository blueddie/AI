from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)

#1 . 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=13, train_size=0.8, stratify=y)

#2 모델
# model = LinearSVC(C=100)        # C가 크면 training포인트를 정확히 구분(굴곡지다), C가 작으면 직선에 가깝다.
model_p = Perceptron()
model_l = LogisticRegression()
model_K = KNeighborsClassifier()
model_D = DecisionTreeClassifier()
model_R = RandomForestClassifier()

models = [model_p, model_l, model_K, model_D, model_R]


for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(model, " acc :", results)  

# Perceptron()  acc : 0.6666666666666666
# LogisticRegression()  acc : 1.0
# KNeighborsClassifier()  acc : 0.9666666666666667
# DecisionTreeClassifier()  acc : 0.9666666666666667
# RandomForestClassifier()  acc : 1.0
from sklearn.datasets import load_diabetes
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_diabetes()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1226, train_size=0.85)

scaler = StandardScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

#2
model = load_model('..\\_data\_save\\MCP\\keras26_diabetes.hdf5')
# #4

loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)

r2 = r2_score(y_test, y_predict)
results = model.predict(X)

# print("R2 score : ", r2)
print("loss : " , loss)

# loss :  71.52172088623047
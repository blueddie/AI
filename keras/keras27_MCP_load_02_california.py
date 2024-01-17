from sklearn.datasets import fetch_california_housing
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.7)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2

#3.
model = load_model('..\\_data\_save\\MCP\\keras26_california.hdf5')

#4.
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)

# R2 score :  0.5729723632802447
# loss :  [0.5123502016067505, 0.0017761989729478955]
from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt


#1. 데이터
datasets = load_diabetes()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1226, train_size=0.8)

#2
model = Sequential()
model.add(Dense(8, input_dim=10))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=90
                   , verbose=1
                   , restore_best_weights=True
                   )
#3
model.compile(loss='mae', optimizer='adam')
start_time = time.time()
hist = model.fit(X_train, y_train, epochs=50, batch_size=3
          , validation_split=0.2
          , callbacks=['es']
          )

end_time = time.time()

#4
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
results = model.predict(X)

print("R2 score : ", r2)
print("loss : " , loss)
print("소요 시간 : ", round(end_time - start_time, 2), "seconds")

print(hist) # wrapping 된 상태
print("========================== hist.history =========================================")
print(hist.history)
print("========================= loss ==============================")
print(hist.history['loss'])
print("========================= val_loss ==============================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')
plt.legend(loc='upper right')
plt.title('diabetes loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  
plt.show()
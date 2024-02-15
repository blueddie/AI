from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ModelCheckpoint

#1 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate([x_train, x_test], axis=0)
y2 = np.concatenate([y_train, y_test], axis=0)

print(np.unique(y2))
y2 = y2.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
y2 = ohe.fit_transform(y2)

print(x.shape)  #(70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
# print(x.shape)      #(70000, 784)
# print(y.shape)      #(70000,)



#2 모델
def make_model(n_component):
    model = Sequential()
    model.add(Dense(32, activation='relu' ,input_shape=(n_component, )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model


n_components = [154, 331, 486, 713, 784]
final_results = []

es = EarlyStopping(monitor='val_accuracy', mode='auto', patience=30, verbose=0, restore_best_weights=True)

for n_component in n_components:
    pca = PCA(n_components=n_component)
    x_pca = pca.fit_transform(x)
    
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_pca, y2, train_size=0.8, random_state=77)
    
    #2 모델
    model = make_model(n_component)
    
    #3 컴파일 , 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    st = time.time()
    model.fit(x_train2, y_train2, batch_size=64, verbose=1, epochs=1000, validation_split=0.2, callbacks=[es])
    et = time.time()
    
    #4 결과
    results = model.evaluate(x_test2, y_test2)
    final_results.append({"n_component" : n_component, "loss" : results[0], "accuracy" : results[1], "time" : round(et - st, 3)})

for result in final_results:
    print(result)

#   4가지 모델
#   1. 70000, 154    
#   2. 70000, 331
#   3. 70000, 486
#   4. 70000, 713
#   시간과 성능을 체크한다.


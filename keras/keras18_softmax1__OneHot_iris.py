import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)

#1 . 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)

X = datasets.data
y = datasets.target

# print(X.shape, y.shape) #(150, 4) (150,)
# print(y.shape)
#-----------------------------------
#pandas
# y = pd.get_dummies(y)
# print(y)

#-------------------
# keras
# from keras.utils import to_categorical
# y_ohe = to_categorical(y)
# y = y_ohe
# print(y)

#----------------------

# sklearn
# oh_enc = OneHotEncoder(sparse=False).fit(y)
# y = oh_enc.transform(y)

#-------------------------
# 선생님
# 111111111111111111



# ohe = ohe.fit(y)
# y_ohe3 = ohe.transform(y)
# print(y_ohe3)


  # fit + transform 대신 쓴다.


print(y.shape)
y = y.reshape(-1, 1) #(150,1)
ohe = OneHotEncoder(sparse=True)
y = ohe.fit_transform(y).toarray() 



#-------------------------------------

# sparse=False 안 쓰면 아래 코드

# print(y.shape)
# oh_enc().fit(y)
# oh_labels = oh_enc.transform(y)
# print(oh_labels)
# print(oh_labels.toarray())
# # print(oh_labels.shape)

# y = oh_labels.toarray()

#----------------

# print(y)
# print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
# print(pd.value_counts(y))

# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13, train_size=0.8, stratify=y)

### 만들어봐

model = Sequential()
model.add(Dense(8, input_dim=4))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(4))
model.add(Dense(3, activation='softmax'))

#3

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=30
                   , verbose=1
                   , restore_best_weights=True
                   )
model.fit(X_train, y_train, epochs=150, batch_size=1
        , validation_split=0.2
        , callbacks=[es]
        )


results = model.evaluate(X_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])


y_predict = model.predict(X_test)

print(y_test)
print(y_predict)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
print(y_test.shape)
print(y_predict.shape)

acc = ACC(y_test, y_predict)
print("accuracy_score : " , acc)










# print("loss : ", loss[0])
# print("accuracy : ", loss[1])

# results = model.predict([[5.9, 3,  5.1, 1.8]])
# print(results)
# max = results[0][0]
# max_index = 0
# for i in range(2):
#     if max < results[0][i + 1] :
#         max = results[0][i + 1]
#         max_index = i + 1 
# results = max_index
# print(results)
# (1, 3)
# results.reshape()
# print(results.shape)    #(1, 3)







# # # [5.1, 3.5, 1.4, 0.2]
# # # [4.9, 3,  1.4, 0.2] 100
# # # [5.9 3.  5.1 1.8] 001

# # # loss :  0.3693692088127136
# # # accuracy :  0.9333333373069763
# # # 1/1 [==============================] - 0s 17ms/step
# # [[0.02058523 0.26513213 0.71428263]]
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE    # anaconda에서 사이킷런 설치할 때 같이 설치됨    없다면  pip install imblearn
import sklearn as sk

#1 데이터
datasets = load_wine()

x = datasets.data
y = datasets['target']
# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(x.shape, y.shape) #(178, 13) (178,)
# print(pd.value_counts(y))
# print(y)

x = x[:-35]
y = y[:-35]
# print(y)
# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71,  8], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=123, stratify=y)

smote = SMOTE(random_state=123)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.value_counts(y_train))


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense

# 2 모델
model = Sequential()
model.add(Dense(15, input_shape=(13,)))
model.add(Dense(16))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(4))
model.add(Dense(3, activation='softmax'))

#3 
model.compile(loss='sparse_categorical_crossentropy' , optimizer='adam', metrics=['accuracy'])

# sparse_categorical_crossentropy -> 원핫 필요하지 않음..  그럼 다른 점은?  taget이 문자라면 label encoding을 통해 역변환을 해야 한다. 결국 뭘 쓰나 똑같다..

model.fit(x_train, y_train, epochs=40, validation_split=0.2)

#4
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ' , results[1])

y_predict = model.predict(x_test)
# print(y_predict)

y_predict = np.argmax(y_predict, axis=1)
# y_predict = y_predict.tolist()

f1 = f1_score(y_test, y_predict, average='macro')
print('f1 : ', f1)



# sparse_categorical_crossentropy 을 했을 때 inverse_transform 하는 방법
# temp = model.predict(x_test)
# temp = np.argmax(temp, axis=1)
# temp = temp.tolist()
# print(temp)

# loss :  0.0881848856806755
# acc :  1.0
# f1 :  1.0

# print("smote 적용")
# from imblearn.over_sampling import SMOTE    # anaconda에서 사이킷런 설치할 때 같이 설치됨    없다면  pip install imblearn
# import sklearn as sk
# print('사이킷 런 ' , sk.__version__)

# SMOTE 할 때 시간 체크해라 너무 오래 걸리면 numpy로 저장해서 써라
# 0    53
# 2    53
# 1    53
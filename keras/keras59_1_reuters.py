from keras.datasets import reuters
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.utils import pad_sequences
from keras.layers import Dense, Conv1D, Flatten, LSTM, Embedding
from keras.models import Sequential



(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=100
                                                         , test_split=0.2)

word_index = reuters.get_word_index()   #30979
print(len(word_index))


print(x_train)
print(x_train.shape, x_test.shape)    #(8982,) (2246,)
print(y_train.shape, y_test.shape)    #(8982,) (2246,)


print(y_train)  #[ 3  4  3 ... 25  3 25]
print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print(len(np.unique(y_train)))  #46
print(len(np.unique(y_test)))  #46

print(type(x_train))    #<class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'>
print(len(x_train[0]), len(x_train[1])) #87 56

# max_len = 0
# for i in range(len(x_train)) :
#     if len(x_train[i]) > max_len :
#         max_len = len(x_train[i])
        
# print(max_len)  #2376

print("뉴스 기사의 최대 길이",max(len(i) for i in x_train)) #2376
print("뉴스 기사의 평균 길이", sum(map(len, x_train)) / len(x_train)) #  145.5398574927633

# 전처리
from keras.utils import pad_sequences

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

# y 원핫은 하고 싶으면 하고 하기 싫으면 sparse_categorical_crossentropy

print(x_train.shape, x_test.shape)  #(8982, 100) (2246, 100)
print(x_train[0].shape) #(100,)


print(y_train.shape, y_test.shape)  #(8982,) (2246,)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
ohe = OneHotEncoder()
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()

print(y_train.shape)    #(8982, 46)
print(y_test.shape)     #(2246, 46)


#2 모델
model  = Sequential()
model.add(Embedding(input_dim=100, output_dim=64, input_length=100))   # 
# input_length 1,5 는 돼, 2,3,4,6 안 돼 모르면 그냥 None 해
# 임베딩 레이어는 벡터화 시킨 다음 3차원 데이터로 출력해준다.
# model.add(Conv1D(97, 3, 64, activation='swish'))
model.add(LSTM(10, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(55, activation='swish'))
model.add(Dense(46, activation='softmax'))
model.summary()

#3
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500, batch_size=168, validation_split=0.2, callbacks=[es])

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


#100 LSTM
# acc :  0.5957257151603699

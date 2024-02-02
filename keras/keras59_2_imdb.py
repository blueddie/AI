from keras.datasets import imdb
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.utils import pad_sequences
from keras.layers import Dense, Conv1D, Flatten, LSTM, Embedding,Bidirectional,  GRU
from keras.models import Sequential



(x_train,y_train), (x_test, y_test) = imdb.load_data(
    num_words=200
)

print(x_train.shape, y_train.shape) #(25000,) (25000,)
print(x_test.shape, y_test.shape)   #(25000,) (25000,)
print(x_train)
print(len(x_train[0]), len(x_test[0]))  #218 68
print(y_train[:20])
print(np.unique(y_train, return_counts=True))

print(type(x_train))    #<class 'numpy.ndarray'>

print("뉴스 기사의 평균 길이", sum(map(len, x_train)) / len(x_train))
# 뉴스 기사의 평균 길이 238.71364

word_len = 238
x_train = pad_sequences(x_train, padding='pre', maxlen=word_len, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=word_len, truncating='pre')

print(x_train.shape, x_test.shape)

#2  모델
model = Sequential()
model.add(Embedding(input_dim=200, output_dim=16, input_length=word_len))
model.add(GRU(12, activation='relu', return_sequences=True))
model.add(Conv1D(16, 2))
model.add(Flatten())
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(55, activation='swish'))
model.add(Dense(1, activation='sigmoid'))
model.summary()



#3
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500, batch_size=250, validation_split=0.2, callbacks=[es])


#4

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# loss :  0.4859594404697418
# acc :  0.7664399743080139

# embedding GRU, Conv1D
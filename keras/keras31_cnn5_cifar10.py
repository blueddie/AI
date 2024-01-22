from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import OneHotEncoder
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
path = '..\\_data\_save\\MCP\\cifar\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'cifar_', date, '_' ,filename])

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=0, save_best_only=True, filepath=filepath)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
# print(np.unique(X_train, return_counts=True))

print(X_train.shape)
print('테스트', X_test.shape)
#---------
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

scaler = StandardScaler()

scaled_train = scaler.fit_transform(X_train_flattened)
scaled_test = scaler.fit_transform(X_test_flattened)
# scaled_array.reshape(original_array.shape)
X_train = scaled_train.reshape(X_train.shape)
X_test = scaled_test.reshape(X_test.shape)
#-----------

print(X_train.shape)
print(X_test.shape)



y_train = OneHotEncoder(sparse=False).fit_transform(y_train)
y_test = OneHotEncoder(sparse=False).fit_transform(y_test)

# Minmax
# X_train = X_train / 255
# X_test = X_test / 255

#-----
model = Sequential()
model.add(Conv2D(97, (3,3), activation='swish', input_shape=(32, 32, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(160, (3,3), activation='swish'))
model.add(Dropout(0.5))
model.add(Conv2D(120, (3,3), activation='swish'))
model.add(GlobalMaxPooling2D())
model.add(Dense(50, activation='swish'))
model.add(Dense(30, activation='swish'))
model.add(Dense(10, activation='softmax'))

es = EarlyStopping(monitor='val_accuracy', mode='auto', patience=20, verbose=0, restore_best_weights=True)
#3.  컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

import time
st = time.time()
model.fit(X_train, y_train, batch_size=64, verbose=1, epochs=110, validation_split=0.2, callbacks=[es, mcp])
et = time.time()

#4. 평가, 예측
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", et - st)

# acc 0.77 이상


# loss :  1.2682442665100098
# acc :  0.5616000294685364
# 걸린 시간 :  118.16578650474548


# loss :  1.0458433628082275
# acc :  0.6391000151634216
# 걸린 시간 :  103.70327425003052


# Strandard Sclaer
# loss :  1.0362398624420166
# acc :  0.6491000056266785
# 걸린 시간 :  119.54423403739929

#-- max pooling
# loss :  0.7828171849250793
# acc :  0.7346000075340271
# 걸린 시간 :  1222.3469178676605


# -- averagpooling2D
# loss :  0.6971752047538757
# acc :  0.763700008392334
# 걸린 시간 :  1126.4281957149506

# loss :  0.7258102297782898
# acc :  0.7649999856948853
# 걸린 시간 :  575.1396014690399

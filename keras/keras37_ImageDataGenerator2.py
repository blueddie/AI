import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D




train_datagen = ImageDataGenerator(
    rescale=1./255
    , horizontal_flip=True      # 수평 뒤집기
    , vertical_flip=True        # 수직 뒤집기
    , width_shift_range=0.2     # 평행 이동하겠다 이동한만큼 0이 채워짐
    , height_shift_range=0.3    # 수직  
    , rotation_range=7          # 정해진 각도만큼 이미지를 회전
    , zoom_range=1.3            # 축소 또는 확대
    , shear_range=0.5           # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    , fill_mode='nearest'       # 빈자리를 가장 비슷한 색으로 채움
    , 
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = 'c:\\_data\\image\\brain\\train\\'
path_test = 'c:\\_data\\image\\brain\\test\\'

st = time.time()
Xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(150, 150)
    , batch_size=160    # 160 이상을 쓰면 X 통으로 가져올 수 있다. batch_size = len(Xy_train)
    , class_mode='binary'
    , shuffle='True'
    # Found 160 images belonging to 2 classes.
)

Xy_test = test_datagen.flow_from_directory(
    path_test
    , target_size=(150, 150)
    , batch_size=120
    , class_mode='binary'
    # , shuffle='True'
    # Found 120 images belonging to 2 classes.
)

et = time.time()
print('이미지 변환에 걸린 시간 : ', round(et - st, 3), ' 초')
print(Xy_train[0][0].shape) #(160, 200, 200, 3) X_train
# print(Xy_train[0][1].shape) #(160,)             y_train

# print(Xy_test[0][0].shape) #(160, 200, 200, 3)  X_test
# print(Xy_test[0][1].shape) #(160,)              y_test


'''
X_train = Xy_train[0][0]
y_train = Xy_train[0][1]

X_test = Xy_test[0][0]
y_test = Xy_test[0][1]


# print(y_train.shape)
# print(y_test.shape)
# print(y_train)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))  # 드롭아웃 추가
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=40, verbose=0, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])


st = time.time()
model.fit(X_train, y_train, batch_size=16, verbose=1, epochs=300, validation_split=0.2, callbacks=[es])
et = time.time()


results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", et - st)

file_acc = str(round(results[1], 4))

model.save('C:\\_data\\_save\\MCP\\brain\\'+ file_acc +'_cnn.hdf5')
# X y 추출해서 모델 만들기
# 성능 0.99이상

# print(type(Xy_train[0][0])) #<class 'numpy.ndarray'> == X_train
# print(type(Xy_train[0][1])) #<class 'numpy.ndarray'> == y_train
# print(type(Xy_test[0][0])) #<class 'numpy.ndarray'> == X_test
# print(type(Xy_test[0][1])) #<class 'numpy.ndarray'> == y_test


# loss :  0.382624089717865
# acc :  0.8333333134651184


'''
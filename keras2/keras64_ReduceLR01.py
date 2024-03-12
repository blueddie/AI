from sklearn.datasets import load_breast_cancer
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import datetime

datasets = load_breast_cancer()

#1. 데이터
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.7, stratify=y)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x.shape)  #(569, 30)

#2 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=30))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# # #3.컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, period=5, filepath="")

rlr = ReduceLROnPlateau(monitor='val_loss',             # 통상 early_stopping patience보다 작다
                        patience=10,
                        mode='min',
                        verbose=1,
                        factor=0.5,
                        # 통상 디폴트보다 높게 잡는다?
                        )
from keras.optimizers import Adam

lr = 0.01
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr))
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es, rlr, mcp])

# #4. 평가, 예측

print('=========================   1. 기본 출력   ====================================')
loss = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0)


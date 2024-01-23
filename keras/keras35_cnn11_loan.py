import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D,Flatten , GlobalMaxPooling2D,GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

csv_path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")


train_csv['근로기간'] = train_csv['근로기간'].str.slice(0, 2)
test_csv['근로기간'] = test_csv['근로기간'].str.slice(0, 2)
train_csv['근로기간'] = train_csv['근로기간'].str.strip()
test_csv['근로기간'] = test_csv['근로기간'].str.strip()
train_csv['근로기간'] = train_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 10})
test_csv['근로기간'] = test_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 10})

# # 근로기간 원핫 인코딩
column_name = '근로기간'
one_hot_encoded = pd.get_dummies(train_csv[column_name], prefix=column_name)    # 근로기간 피처에 대해 원핫인코딩을 수행합니다.
train_csv = pd.concat([train_csv, one_hot_encoded], axis=1) # 기존 데이터프레임에 원핫인코딩된 결과를 추가합니다.
train_csv.drop(column_name, axis=1, inplace=True)   # 기존 근로기간 컬럼을 삭제합니다.

one_hot_encoded = pd.get_dummies(test_csv[column_name], prefix=column_name)    # 근로기간 피처에 대해 원핫인코딩을 수행합니다.
test_csv = pd.concat([test_csv, one_hot_encoded], axis=1) # 기존 데이터프레임에 원핫인코딩된 결과를 추가합니다.
test_csv.drop(column_name, axis=1, inplace=True)   # 기존 근로기간 컬럼을 삭제합니다.

# 대출기간 처리 ohe
column_name = '대출기간'
one_hot_encoded = pd.get_dummies(train_csv[column_name], prefix=column_name)
train_csv = pd.concat([train_csv, one_hot_encoded], axis=1)
train_csv.drop(column_name, axis=1, inplace=True)

one_hot_encoded = pd.get_dummies(test_csv[column_name], prefix=column_name)
test_csv = pd.concat([test_csv, one_hot_encoded], axis=1)
test_csv.drop(column_name, axis=1, inplace=True)

# 주택 소유 상태
train_csv['주택소유상태'] = train_csv['주택소유상태'].replace({'ANY' : 'MORTGAGE'})
column_name = '주택소유상태'
one_hot_encoded = pd.get_dummies(train_csv[column_name], prefix=column_name)
train_csv = pd.concat([train_csv, one_hot_encoded], axis=1)
train_csv.drop(column_name, axis=1, inplace=True)

one_hot_encoded = pd.get_dummies(test_csv[column_name], prefix=column_name)
test_csv = pd.concat([test_csv, one_hot_encoded], axis=1)
test_csv.drop(column_name, axis=1, inplace=True)


# 대출목적
test_csv['대출목적'] = test_csv['대출목적'].replace({'결혼' : '부채 통합'})
column_name = '대출목적'
one_hot_encoded = pd.get_dummies(train_csv[column_name], prefix=column_name)
train_csv = pd.concat([train_csv, one_hot_encoded], axis=1)
train_csv.drop(column_name, axis=1, inplace=True)

one_hot_encoded = pd.get_dummies(test_csv[column_name], prefix=column_name)
test_csv = pd.concat([test_csv, one_hot_encoded], axis=1)
test_csv.drop(column_name, axis=1, inplace=True)

columns_to_drop = ['대출등급','연체계좌수','총연체금액']

X = train_csv.drop(columns=columns_to_drop)
y = train_csv['대출등급']

test_drop = ['연체계좌수', '총연체금액']
test_csv = test_csv.drop(columns=test_drop)


X = np.asarray(X).astype(np.float32) 
test_csv = np.asarray(test_csv).astype(np.float32)


print(X.shape)  #(96294, 35)
print(test_csv.shape)   #(64197, 35)

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
# print(y.shape)

y = y.reshape(-1, 1)
print('y :', y.shape)

X = X.reshape(-1, 5, 7, 1)
test_csv = test_csv.reshape(-1, 5, 7, 1)
print(X.shape)  #(96294, 5, 7, 1)
print(test_csv.shape)   #(64197, 5, 7, 1)

ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y = ohe.transform(y)
# print('원핫 : ', y)



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=112847)

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)
test_csv_flattened = test_csv.reshape(test_csv.shape[0], -1)

scaler = StandardScaler()
scaler.fit(X_train_flattened)
scaled_train = scaler.transform(X_train_flattened)
scaled_test = scaler.transform(X_test_flattened)
scaled_test_csv = scaler.transform(test_csv_flattened)

X_train = scaled_train.reshape(X_train.shape)
X_test = scaled_test.reshape(X_test.shape)
test_csv = scaled_test_csv.reshape(test_csv.shape)

#model = Sequential()
model = Sequential()
model.add(Conv2D(120, (3,3), activation='relu', padding='same', input_shape=(5, 7, 1)))
model.add(MaxPooling2D(strides=(1, 1)))
model.add(Conv2D(90, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(strides=(1, 1)))
model.add(Conv2D(141, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(strides=(1, 1)))
model.add(Conv2D(77, (3,3), activation='relu', padding='same'))
model.add(GlobalMaxPooling2D())
model.add(Dense(97, activation='relu'))
# model.add(Dropout(0.3)) # 방금 추가
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='softmax')) 

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy'
                , mode='max'
                , patience=50
                , verbose=1
                , restore_best_weights=True
                )

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
path = 'C:\\_data\\_save\\MCP\\dacon_loan\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'loan_test', date, '_' ,filename])



#3 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=1000, batch_size=512, validation_split=0.24, callbacks=[es])

#4
results = model.evaluate(X_test, y_test)
acc = results[1]
loss = results[0]

y_predict = model.predict(X_test)
y_predict = np.argmax(y_predict, axis=1)
y_predict = encoder.inverse_transform(y_predict)

y_test = np.argmax(y_test, axis=1)
y_test = encoder.inverse_transform(y_test)




f1 = f1_score(y_test, y_predict, average='macro')

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis=1)
y_submit = encoder.inverse_transform(y_submit)

submission_csv['대출등급'] = y_submit

print('acc : ' , results[1])
print('f1 : ' , f1)


model.save('C:\\_data\\_save\\MCP\\dacon_loan\\0123\\best_1_cnn.hdf5')

submission_csv.to_csv(csv_path + date + "_f1_" + str(f1) + ".csv", index=False)

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D,Flatten , GlobalMaxPooling2D,GlobalAveragePooling2D, GlobalMaxPooling1D, Conv1D
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


xy = train_csv
#################################

unknown_replacement = xy['근로기간'].mode()[0]
xy.loc[xy['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement
test_csv.loc[test_csv['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement

# xy['근로기간'] = xy['근로기간'].str.slice(0, 2)
# xy['근로기간'] = xy['근로기간'].str.strip()
# xy['근로기간'] = xy['근로기간'].replace({'<' : 0, '<1' :0})
xy.loc[xy['근로기간'] == '<1 year', '근로기간'] = '< 1 year'
xy.loc[xy['근로기간'] == '3', '근로기간'] = '3 years'
xy.loc[xy['근로기간'] == '10+years', '근로기간'] = '10+ years'
xy.loc[xy['근로기간'] == '1 years', '근로기간'] = '1 year'

test_csv.loc[test_csv['근로기간'] == '<1 year', '근로기간'] = '< 1 year'
test_csv.loc[test_csv['근로기간'] == '3', '근로기간'] = '3 years'
test_csv.loc[test_csv['근로기간'] == '10+years', '근로기간'] = '10+ years'
test_csv.loc[test_csv['근로기간'] == '1 years', '근로기간'] = '1 year'

# test_csv.loc[test_csv['근로기간'] == '< 1 year', '근로기간'] = '0'
# test_csv.loc[test_csv['근로기간'] == '2 years', '근로기간'] = '1'
# test_csv.loc[test_csv['근로기간'] == '3 years', '근로기간'] = '0'
# test_csv.loc[test_csv['근로기간'] == '4 years', '근로기간'] = '0'
# test_csv.loc[test_csv['근로기간'] == '5 years', '근로기간'] = '0'
# test_csv.loc[test_csv['근로기간'] == '6 years', '근로기간'] = '0'
# test_csv.loc[test_csv['근로기간'] == '7 years', '근로기간'] = '0'
# test_csv.loc[test_csv['근로기간'] == '8 years', '근로기간'] = '0'
# test_csv.loc[test_csv['근로기간'] == '9 years', '근로기간'] = '0'
# test_csv.loc[test_csv['근로기간'] == '10 years', '근로기간'] = '2'

# xy.loc[xy['근로기간'] == '< 1 year', '근로기간'] = '0'
# xy.loc[xy['근로기간'] == '2 years', '근로기간'] = '1'
# xy.loc[xy['근로기간'] == '3 years', '근로기간'] = '0'
# xy.loc[xy['근로기간'] == '4 years', '근로기간'] = '0'
# xy.loc[xy['근로기간'] == '5 years', '근로기간'] = '0'
# xy.loc[xy['근로기간'] == '6 years', '근로기간'] = '0'
# xy.loc[xy['근로기간'] == '7 years', '근로기간'] = '0'
# xy.loc[xy['근로기간'] == '8 years', '근로기간'] = '0'
# xy.loc[xy['근로기간'] == '9 years', '근로기간'] = '0'
# xy.loc[xy['근로기간'] == '10 years', '근로기간'] = '2'

########################################
# xy.loc[xy['대출목적'] == '주택 개선', '대출목적'] = '0'
# xy.loc[xy['대출목적'] == '주요 구매', '대출목적'] = '0'
# xy.loc[xy['대출목적'] == '휴가', '대출목적'] = '0'
# xy.loc[xy['대출목적'] == '의료', '대출목적'] = '0'
# xy.loc[xy['대출목적'] == '자동차', '대출목적'] = '0'
# xy.loc[xy['대출목적'] == '신용 카드', '대출목적'] = '1'
# xy.loc[xy['대출목적'] == '소규모 사업', '대출목적'] = '0'
# xy.loc[xy['대출목적'] == '기타', '대출목적'] = '0'
# xy.loc[xy['대출목적'] == '이사', '대출목적'] = '0'
# xy.loc[xy['대출목적'] == '주택', '대출목적'] = '0'
# xy.loc[xy['대출목적'] == '재생 에너지', '대출목적'] = '0'
# xy.loc[xy['대출목적'] == '부채통합', '대출목적'] = '2'

# test_csv.loc[test_csv['대출목적'] == '주택 개선', '대출목적'] = '0'
# test_csv.loc[test_csv['대출목적'] == '주요 구매', '대출목적'] = '0'
# test_csv.loc[test_csv['대출목적'] == '휴가', '대출목적'] = '0'
# test_csv.loc[test_csv['대출목적'] == '의료', '대출목적'] = '0'
# test_csv.loc[test_csv['대출목적'] == '자동차', '대출목적'] = '0'
# test_csv.loc[test_csv['대출목적'] == '신용 카드', '대출목적'] = '1'
# test_csv.loc[test_csv['대출목적'] == '소규모 사업', '대출목적'] = '0'   
# test_csv.loc[test_csv['대출목적'] == '기타', '대출목적'] = '0'
# test_csv.loc[test_csv['대출목적'] == '이사', '대출목적'] = '0'
# test_csv.loc[test_csv['대출목적'] == '주택', '대출목적'] = '0'
# test_csv.loc[test_csv['대출목적'] == '재생 에너지', '대출목적'] = '0'
# test_csv.loc[test_csv['대출목적'] == '부채통합', '대출목적'] = '2'

encoder = LabelEncoder()
encoder.fit(xy['근로기간'])
xy['근로기간'] = encoder.transform(xy['근로기간'])
test_csv['근로기간'] = encoder.transform(test_csv['근로기간'])
# print(np.unique(test_csv['근로기간'], return_counts=True))
# print(pd.value_counts(xy['근로기간']))

# 대출 기간
encoder = LabelEncoder()
encoder.fit(xy['대출기간'])
xy['대출기간'] = encoder.transform(xy['대출기간'])
test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])

# 주택소유상태
xy.loc[xy['주택소유상태'] == 'ANY', '주택소유상태'] = 'MORTGAGE'

encoder.fit(xy['주택소유상태'])
xy['주택소유상태'] = encoder.transform(xy['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])
# test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])
# print(np.unique(xy['주택소유상태']))
# print(np.unique(test_csv['주택소유상태']))


#대출 목적
# print(np.unique(xy['대출목적']))
# print(np.unique(test_csv['대출목적']))
test_csv.loc[test_csv['대출목적'] == '결혼', '대출목적'] = '부채 통합'


encoder.fit(xy['대출목적'])
xy['대출목적'] = encoder.transform(xy['대출목적'])
test_csv['대출목적'] = encoder.transform(test_csv['대출목적'])
# print(np.unique(xy['대출목적']))
# print(np.unique(test_csv['대출목적']))
# print(xy.shape) #(90293, 14)

columns_to_drop = ['대출등급']
x = xy.drop(columns=columns_to_drop)

# columns_to_drop_test = ['연체계좌수']
# test_csv = test_csv.drop(columns=columns_to_drop_test)
# print(x.shape)
# x = x.astype(np.float32)

print(x.info())
y = xy['대출등급']

# print(y.shape)  #(90293,)
y = y.values.reshape(-1, 1)
# print(y.shape)
ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y = ohe.transform(y)

print(x.shape)
x = x.values.reshape(-1, 13, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.84, random_state=112847)

X_train_flattened = X_train.values.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)
test_csv_flattened = test_csv.reshape(test_csv.shape[0], -1)

scaler = MinMaxScaler()
scaler.fit(X_train_flattened)
scaled_train = scaler.transform(X_train_flattened)
scaled_test = scaler.transform(X_test_flattened)
scaled_test_csv = scaler.transform(test_csv_flattened)

X_train = scaled_train.reshape(X_train.shape)
X_test = scaled_test.reshape(X_test.shape)
test_csv = scaled_test_csv.reshape(test_csv.shape)

#model = Sequential()
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(13, 1)))
model.add(GlobalMaxPooling1D())
model.add(Conv1D(16))
model.add(GlobalMaxPooling1D())
model.add(Conv1D(16))
model.add(Flatten())
model.add(Dense(97, activation='relu'))
# model.add(Dropout(0.3)) # 방금 추가
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='softmax')) 

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy'
                , mode='max'
                , patience=100
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

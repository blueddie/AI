import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE    # anaconda에서 사이킷런 설치할 때 같이 설치됨    없다면  pip install imblearn


csv_path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

encoder = LabelEncoder()
# ohe = OneHotEncoder()

train_csv['근로기간'] = train_csv['근로기간'].str.slice(0, 2)
test_csv['근로기간'] = test_csv['근로기간'].str.slice(0, 2)
train_csv['근로기간'] = train_csv['근로기간'].str.strip()
test_csv['근로기간'] = test_csv['근로기간'].str.strip()
train_csv['근로기간'] = train_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 10})
test_csv['근로기간'] = test_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 10})

encoder.fit(train_csv['대출기간'])
train_csv['대출기간'] = encoder.transform(train_csv['대출기간'])
test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])


train_csv['주택소유상태'] = train_csv['주택소유상태'].replace({'ANY' : 'MORTGAGE'})
train_csv['주택소유상태'] = train_csv['주택소유상태'].replace({'MORTGAGE' : 3, 'OWN' : 2, 'RENT' : 0})
test_csv['주택소유상태'] = test_csv['주택소유상태'].replace({'MORTGAGE' : 3, 'OWN' : 2, 'RENT' : 0})

encoder.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = encoder.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])

test_csv['대출목적'] = test_csv['대출목적'].replace({'결혼' : '부채 통합'})

train_csv['대출목적'] = train_csv['대출목적'].replace({'부채 통합' : 0, '주택 개선' : 1, '주요 구매' : 2, '휴가' : 3, '의료' : 4, '자동차' : 5, '신용 카드' : 6, '소규모 사업' : 7, '기타' : 8, '이사' : 9 ,'주택': 10,'재생 에너지': 11})
test_csv['대출목적'] = test_csv['대출목적'].replace({'부채 통합' : 0, '주택 개선' : 1, '주요 구매' : 2, '휴가' : 3, '의료' : 4, '자동차' : 5, '신용 카드' : 6, '소규모 사업' : 7, '기타' : 8, '이사' : 9 ,'주택': 10,'재생 에너지': 11})

columns_to_drop = ['대출등급']
x = train_csv.drop(columns=columns_to_drop)
y = train_csv['대출등급']

# print(x.shape)  #(96294, 13)
# print(y.shape)  #(96294,)


y = y.values.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y = ohe.transform(y)#(96294, 7)
print(y.shape)




x_train, x_test, y_train, y_test = train_test_split(x, y ,random_state=1555, train_size=0.75, stratify=y)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

smote = SMOTE(random_state=155)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(len(x_train))




# print(pd.value_counts(y_train))


#2
model = Sequential()
model.add(Dense(212, activation='swish', input_shape=(13,)))
model.add(Dense(150, activation='swish'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(231, activation='swish'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(354, activation='swish'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(210, activation='swish'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(164, activation='swish'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(115, activation='swish'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(225, activation='swish'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(125, activation='swish'))
model.add(Dense(7, activation='softmax')) 
model.summary()


# x_train, x_test, y_train, y_test = train_test_split(x, y ,random_state=123, train_size=0.8, stratify=y)

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss'
                , mode='min'
                , patience=300
                , verbose=1
                , restore_best_weights=True
                )

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
path = 'C:\\_data\\_save\\MCP\\dacon_loan\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'loan_test', date, '_' ,filename])

#3
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=40000, batch_size=1024, validation_split=0.3, callbacks=[es])

#4
results = model.evaluate(x_test, y_test)
acc = results[1]
loss = results[0]

y_predict = model.predict(x_test)   
y_predict = ohe.inverse_transform(y_predict)

y_test = ohe.inverse_transform(y_test)

f1 = f1_score(y_test, y_predict, average='macro')

y_submit = model.predict(test_csv)
# y_submit = np.argmax(y_submit, axis=1)
y_submit = ohe.inverse_transform(y_submit)

submission_csv['대출등급'] = pd.DataFrame(y_submit.reshape(-1,1))

print("acc : " , acc)
print('f1 : ', f1)

model.save('C:\\_data\\_save\\models\\loan\\0126\\'+ date +'best_dnn.hdf5')
submission_csv.to_csv(csv_path + date + "_f1_" + str(f1) + ".csv", index=False)



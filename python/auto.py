import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.preprocessing import MinMaxScaler ,StandardScaler, RobustScaler, Normalizer
import datetime, time
import matplotlib.pyplot as plt


csv_path = "C:\\_data\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

# 1. 데이터



pd.set_option('display.max_rows', None)

encoder = LabelEncoder()



train_str = train_csv['대출기간'].str.slice(0, 3)
test_str = test_csv['대출기간'].str.slice(0, 3)

train_csv['대출기간'] = pd.to_numeric(train_str)
test_csv['대출기간'] = pd.to_numeric(test_str)

train_csv['대출기간'] = encoder.fit_transform(train_csv['대출기간'])
test_csv['대출기간'] = encoder.fit_transform(test_csv['대출기간'])

train_csv['대출금액'] = train_csv['대출금액'] / 10000
test_csv['대출금액'] = test_csv['대출금액'] / 10000

train_csv['근로기간'] = train_csv['근로기간'].str.slice(0, 2)
test_csv['근로기간'] = test_csv['근로기간'].str.slice(0, 2)
train_csv['근로기간'] = train_csv['근로기간'].str.strip()
test_csv['근로기간'] = test_csv['근로기간'].str.strip()
train_csv['근로기간'] = train_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 5})
test_csv['근로기간'] = test_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 5})


idx = train_csv[train_csv['주택소유상태'] == 'ANY'].index
train_csv = train_csv.drop(idx)

train_csv['주택소유상태'] = encoder.fit_transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = encoder.fit_transform(test_csv['주택소유상태'])

# train_csv['연간소득'] = encoder.fit_transform(train_csv['연간소득'] / 1000)
# test_csv['연간소득'] = encoder.fit_transform(test_csv['연간소득'] / 1000) 

train_csv['연간소득'] = train_csv['연간소득'] / 1000
test_csv['연간소득'] = test_csv['연간소득'] / 1000

# 대출목적 test 결혼 replace
test_csv['대출목적'] = test_csv['대출목적'].str.replace('결혼', '부채통합')


train_csv['대출목적'] = np.where(train_csv['대출목적'] == '부채 통합' , 0, train_csv['대출목적'])
train_csv['대출목적'] = np.where(train_csv['대출목적'] == '신용 카드' , 1, train_csv['대출목적'])
train_csv['대출목적'] = np.where((train_csv['대출목적'] != 0) & (train_csv['대출목적'] != 1 ), 2, train_csv['대출목적'])


test_csv['대출목적'] = np.where(test_csv['대출목적'] == '부채 통합' , 0, test_csv['대출목적'])
test_csv['대출목적'] = np.where(test_csv['대출목적'] == '신용 카드' , 1, test_csv['대출목적'])
test_csv['대출목적'] = np.where((test_csv['대출목적'] != 0) & (test_csv['대출목적'] != 1 ), 2, test_csv['대출목적'])

encoder.fit(train_csv['대출등급'])
X = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']


sds = StandardScaler()
mms = MinMaxScaler()

y = y.values.reshape(-1, 1)
y = OneHotEncoder(sparse=False).fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=3, train_size=0.86, stratify=y)
#2 
input1 = Input(shape=(13,))
dense1 = Dense(19, activation='relu')(input1)
dense2 = Dense(97, activation='relu')(dense1)
dense3 = Dense(9, activation='relu')(dense2)
dense4 = Dense(21, activation='relu')(dense3)
dense5 = Dense(16, activation='relu')(dense4)
dense6 = Dense(21, activation='relu')(dense5)
output1 = Dense(7, activation='softmax')(dense6)

model = Model(inputs=input1, outputs=output1)




date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
path = '..\\_data\_save\\MCP\\dacon_loan\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'loan_test', date, '_' ,filename])

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=1000, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

mms = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=3, train_size=0.83, stratify=y)

#---mms
mms.fit(X_train)

X_train = mms.transform(X_train)
X_test = mms.transform(X_test)
test_csv = mms.transform(test_csv)



# -------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

st = time.time()
hist = model.fit(X_train, y_train, epochs=20000, batch_size=555, validation_split=0.2, callbacks=[es, mcp])
et = time.time()



results = model.evaluate(X_test, y_test)
acc = results[1]
loss = results[0]

y_predict = model.predict(X_test)
y_predict = np.argmax(y_predict, axis=1)
y_predict = encoder.inverse_transform(y_predict)
y_test = np.argmax(y_test, axis=1)
y_test = encoder.inverse_transform(y_test)

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis=1)
y_submit = encoder.inverse_transform(y_submit)

f1 = f1_score(y_test, y_predict, average='macro')

submission_csv['대출등급'] = y_submit

submission_csv.to_csv(csv_path + date + '전처리테스트' + str(round(f1, 5)) + ".csv", index=False)
print("걸린 시간 :" , et - st)
print('f1 : ' , f1)
print('loss : ' , loss)
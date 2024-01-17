import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

encoder = LabelEncoder()

encoder.fit(train_csv['주택소유상태'])

# print(pd.value_counts(train_csv['주택소유상태']))
# print('인코당 클래스: ', encoder.classes_)  # ['ANY' 'MORTGAGE' 'OWN' 'RENT']

# print(pd.value_counts(test_csv['주택소유상태']))
# print('인코당 클래스: ', encoder.classes_)  # ['MORTGAGE' 'OWN' 'RENT']

idx_any = train_csv[train_csv['주택소유상태'] == 'ANY'].index
train_csv = train_csv.drop(idx_any) # ANY가 있는 행 제거

train_csv['주택소유상태'] = encoder.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])


train_csv['대출목적'] = train_csv['대출목적'].str.replace(' ' , '')
test_csv['대출목적'] = test_csv['대출목적'].str.replace(' ' , '')

encoder.fit(train_csv['대출목적'])
train_csv['대출목적'] = encoder.transform(train_csv['대출목적'])

# f['B'].str.replace('데이터,', '소프트웨어')
test_csv['대출목적'] = test_csv['대출목적'].str.replace('결혼', '부채통합')

test_csv['대출목적'] = encoder.transform(test_csv['대출목적'])
# print(pd.value_counts(test_csv['대출목적']))

# print('인코당 클래스: ', encoder.classes_)  #인코당 클래스:  ['기타' '부채통합' '소규모사업' '신용카드' '의료' '이사' '자동차' '재생에너지' '주요구매' '주택' '주택개선' '휴가']

# df['A'].str.slice(1, 5)
 
train_str = train_csv['대출기간'].str.slice(0, 3)
test_str = test_csv['대출기간'].str.slice(0, 3)

train_csv['대출기간'] = pd.to_numeric(train_str)
test_csv['대출기간'] = pd.to_numeric(test_str)

encoder.fit(train_csv['대출기간'])
train_csv['대출기간'] = encoder.transform(train_csv['대출기간'])
test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])

train_csv['근로기간'] = train_csv['근로기간'].str.slice(0, 2)
test_csv['근로기간'] = test_csv['근로기간'].str.slice(0, 2)
train_csv['근로기간'] = train_csv['근로기간'].str.strip()
test_csv['근로기간'] = test_csv['근로기간'].str.strip()
# print(pd.value_counts(test_csv['근로기간']))
train_csv['근로기간'] = train_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 11})
test_csv['근로기간'] = test_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 11})



# encoder.fit(train_csv['대출금액'])
# train_csv['대출금액'] = encoder.transform(train_csv['대출금액'])
# test_csv['대출금액'] = encoder.transform(test_csv['대출금액'])
# print(train_csv['대출금액'])


# print(pd.value_counts(train_csv['대출금액']))


# print(pd.value_counts(test_csv['근로기간']))

# print(pd.value_counts(test_csv['근로기간']))



encoder.fit(train_csv['대출등급'])

X = train_csv.drop(['대출등급'], axis=1)

y = train_csv['대출등급']

# y = pd.get_dummies(y, dtype='int')
# # print(y.shape)  #(96293, 7)
# print(y)  #(96293, 7)
# # print(y.idxmax(axis=1))

# ------ mms
# mms = MinMaxScaler()
# mms.fit(X)
# X = mms.transform(X)
# test_csv = mms.transform(test_csv)

#-------- sklearn
y = y.values.reshape(-1, 1)
y = OneHotEncoder(sparse=False).fit_transform(y)


# print(y.shape)
# ohe = OneHotEncoder(sparse=True)
# y = ohe.fit_transform(y).toarray() 



# print(y.shape)  #(96294, 7)
# print(X)
# print(y)

# ------------

#2
model = Sequential()
model.add(Dense(19, activation='relu', input_shape=(13,)))
model.add(Dense(97,activation='relu' ))
model.add(Dense(9,activation='relu' ))
model.add(Dense(21,activation='relu' ))
model.add(Dense(16,activation='relu' ))
model.add(Dense(21, activation='relu'))
model.add(Dense(7, activation='softmax'))

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='accuracy'
                   , mode='max'
                   , patience=1000
                   , verbose=1
                   , restore_best_weights=True
                   )

mms = MinMaxScaler()
sds = StandardScaler()

def auto(test_csv) :
    rs = random.randrange(2, 99999999)
    # bs = random.randrange(5000, 11999)
    X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=rs, train_size=0.89, stratify=y)
    
    #---mms
    mms.fit(X_train)
    X_train = mms.transform(X_train)
    X_test = mms.transform(X_test)
    test_csv = mms.transform(test_csv)

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='..\\_data\\_save\\MCP\\keras26_loan_1.hdf5')

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, epochs=50000, batch_size=500, validation_split=0.14, callbacks=[es, mcp])
    


    results = model.evaluate(X_test, y_test)
    acc = results[1]
    loss = results[0]
    
    y_predict = model.predict(X_test)
    # print(y_predict)
    y_predict = np.argmax(y_predict, axis=1)
    y_predict = encoder.inverse_transform(y_predict)
    # print("예측", y_predict)
    y_test = np.argmax(y_test, axis=1)
    y_test = encoder.inverse_transform(y_test)
    # print(y_test)


    y_submit = model.predict(test_csv)
    y_submit = np.argmax(y_submit, axis=1)
    y_submit = encoder.inverse_transform(y_submit)
    # print(pd.value_counts(y_test))
    # y_predict = encoder.inverse_transform(y_test)
    # print(pd.value_counts(y_predict))
    # print(pd.value_counts(y_submit))
    submission_csv['대출등급'] = y_submit
    f1 = f1_score(y_test, y_predict, average='macro')
    # submission_csv.to_csv(path + "0115_" + str(rs) + "_bs_" + str(bs) + "f1_" + str(round(f1, 3)) + ".csv", index=False)
    return f1, rs, hist


# f1, rs, bs , hist = auto()
# print("f1 : " , f1)

max_f1 = 0.936

while True:
    f1, rs, hist = auto(test_csv)
    if f1 > max_f1 :
        max_f1 = f1
        submission_csv.to_csv(path + "0117_936_mac_" + str(rs) + "_f1_" + str(f1) + ".csv", index=False)
        break
# print(f1)
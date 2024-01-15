import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random

path = "C:\\_data\dacon\\loan_grade\\"

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

train_csv['근로기간'] = train_csv['근로기간'].str.slice(0, 2)
test_csv['근로기간'] = test_csv['근로기간'].str.slice(0, 2)
train_csv['근로기간'] = train_csv['근로기간'].str.strip()
test_csv['근로기간'] = test_csv['근로기간'].str.strip()
# print(pd.value_counts(test_csv['근로기간']))
train_csv['근로기간'] = train_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 11})
test_csv['근로기간'] = test_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 11})
# print(pd.value_counts(test_csv['근로기간']))

# print(pd.value_counts(test_csv['근로기간']))

encoder.fit(train_csv['대출등급'])

X = train_csv.drop(['대출등급'], axis=1)

y = train_csv['대출등급']

# y = pd.get_dummies(y, dtype='int')
# # print(y.shape)  #(96293, 7)
# print(y)  #(96293, 7)
# # print(y.idxmax(axis=1))


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
model.add(Dense(64, input_shape=(13,)))
model.add(Dense(32))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(7, activation='softmax'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=150
                   , verbose=1
                   , restore_best_weights=True
                   )


def auto() :
    rs = random.randrange(1, 99999999)
    bs = random.randrange(256, 1025)
    X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=rs, train_size=0.85, stratify=y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, epochs=3000, batch_size=bs, validation_split=0.2, callbacks=[es])
    
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
    return f1, rs, bs, hist


# f1, rs, bs = auto()
# print("f1 : " , f1)

max_f1 = 0.6

while True:
    f1, rs, bs, hist = auto()
    if f1 > max_f1 :
        max_f1 = f1
        submission_csv.to_csv(path + "0115_ov6_" + str(rs) + "_bs_" + str(bs) + "f1_" + str(f1) + ".csv", index=False)
print(f1)

f1, rs, bs, hist = auto()



# import matplotlib.pyplot as plt
# # plt.rcParams['font.family'] = 'Malgun Gothic'   #한글 깨짐 해결
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')
# plt.legend(loc='upper right')
# plt.title('grade')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()  
# plt.show()
    

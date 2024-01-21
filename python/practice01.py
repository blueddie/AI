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

csv_path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

# encoder = LabelEncoder()

# encoder.fit(train_csv['주택소유상태'])
# idx_any = train_csv[train_csv['주택소유상태'] == 'ANY'].index
# train_csv = train_csv.drop(idx_any) # ANY가 있는 행 제거

# train_csv['주택소유상태'] = encoder.transform(train_csv['주택소유상태'])
# test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])

# train_csv['대출목적'] = train_csv['대출목적'].str.replace(' ' , '')
# test_csv['대출목적'] = test_csv['대출목적'].str.replace(' ' , '')

# encoder.fit(train_csv['대출목적'])
# train_csv['대출목적'] = encoder.transform(train_csv['대출목적'])
# test_csv['대출목적'] = test_csv['대출목적'].str.replace('결혼', '부채통합')
# test_csv['대출목적'] = encoder.transform(test_csv['대출목적'])
 
# train_str = train_csv['대출기간'].str.slice(0, 3)
# test_str = test_csv['대출기간'].str.slice(0, 3)

# train_csv['대출기간'] = pd.to_numeric(train_str)
# test_csv['대출기간'] = pd.to_numeric(test_str)

# encoder.fit(train_csv['대출기간'])
# train_csv['대출기간'] = encoder.transform(train_csv['대출기간'])
# test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])

# train_csv['근로기간'] = train_csv['근로기간'].str.slice(0, 2)
# test_csv['근로기간'] = test_csv['근로기간'].str.slice(0, 2)
# train_csv['근로기간'] = train_csv['근로기간'].str.strip()
# test_csv['근로기간'] = test_csv['근로기간'].str.strip()
# # print(pd.value_counts(test_csv['근로기간']))
# train_csv['근로기간'] = train_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 11})
# test_csv['근로기간'] = test_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 11})

# encoder.fit(train_csv['대출등급'])
# X = train_csv.drop(['대출등급'], axis=1)
# y = train_csv['대출등급']



#------------

encoder = LabelEncoder()



# train_str = train_csv['대출기간'].str.slice(0, 3)
# test_str = test_csv['대출기간'].str.slice(0, 3)

# train_csv['대출기간'] = pd.to_numeric(train_str)
# test_csv['대출기간'] = pd.to_numeric(test_str)

# train_csv['대출기간'] = encoder.fit_transform(train_csv['대출기간'])
# test_csv['대출기간'] = encoder.fit_transform(test_csv['대출기간'])

# train_csv['대출금액'] = train_csv['대출금액'] / 10000
# test_csv['대출금액'] = test_csv['대출금액'] / 10000

# train_csv['근로기간'] = train_csv['근로기간'].str.slice(0, 2)
# test_csv['근로기간'] = test_csv['근로기간'].str.slice(0, 2)
# train_csv['근로기간'] = train_csv['근로기간'].str.strip()
# test_csv['근로기간'] = test_csv['근로기간'].str.strip()
# train_csv['근로기간'] = train_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 5})
# test_csv['근로기간'] = test_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 5})


# idx = train_csv[train_csv['주택소유상태'] == 'ANY'].index
# train_csv = train_csv.drop(idx)

# train_csv['주택소유상태'] = encoder.fit_transform(train_csv['주택소유상태'])
# test_csv['주택소유상태'] = encoder.fit_transform(test_csv['주택소유상태'])

# # train_csv['연간소득'] = encoder.fit_transform(train_csv['연간소득'] / 1000)
# # test_csv['연간소득'] = encoder.fit_transform(test_csv['연간소득'] / 1000) 

# train_csv['연간소득'] = train_csv['연간소득'] / 1000
# test_csv['연간소득'] = test_csv['연간소득'] / 1000

# # 대출목적 test 결혼 replace
# test_csv['대출목적'] = test_csv['대출목적'].str.replace('결혼', '부채통합')


# train_csv['대출목적'] = np.where(train_csv['대출목적'] == '부채 통합' , 0, train_csv['대출목적'])
# train_csv['대출목적'] = np.where(train_csv['대출목적'] == '신용 카드' , 1, train_csv['대출목적'])
# train_csv['대출목적'] = np.where((train_csv['대출목적'] != 0) & (train_csv['대출목적'] != 1 ), 2, train_csv['대출목적'])


# test_csv['대출목적'] = np.where(test_csv['대출목적'] == '부채 통합' , 0, test_csv['대출목적'])
# test_csv['대출목적'] = np.where(test_csv['대출목적'] == '신용 카드' , 1, test_csv['대출목적'])
# test_csv['대출목적'] = np.where((test_csv['대출목적'] != 0) & (test_csv['대출목적'] != 1 ), 2, test_csv['대출목적'])

# encoder.fit(train_csv['대출등급'])
# X = train_csv.drop(['대출등급'], axis=1)
# y = train_csv['대출등급']

#-------------------------
train_csv['근로기간'] = train_csv['근로기간'].str.slice(0, 2)
test_csv['근로기간'] = test_csv['근로기간'].str.slice(0, 2)
train_csv['근로기간'] = train_csv['근로기간'].str.strip()
test_csv['근로기간'] = test_csv['근로기간'].str.strip()
train_csv['근로기간'] = train_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 10})
test_csv['근로기간'] = test_csv['근로기간'].replace({'<' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, '10' : 10, '<1' : 0, 'Un' : 10})

# 근로기간 라벨 인코딩
encoder.fit(train_csv['근로기간'])
train_csv['근로기간'] = encoder.transform(train_csv['근로기간'])
test_csv['근로기간'] = encoder.transform(test_csv['근로기간'])

# 근로기간 원핫 인코딩
column_name = '근로기간'
# one_hot_encoded = pd.get_dummies(train_csv[column_name], prefix=column_name)    # 근로기간 피처에 대해 원핫인코딩을 수행합니다.
# train_csv = pd.concat([train_csv, one_hot_encoded], axis=1) # 기존 데이터프레임에 원핫인코딩된 결과를 추가합니다.
# train_csv.drop(column_name, axis=1, inplace=True)   # 기존 근로기간 컬럼을 삭제합니다.

# one_hot_encoded = pd.get_dummies(test_csv[column_name], prefix=column_name)    # 근로기간 피처에 대해 원핫인코딩을 수행합니다.
# test_csv = pd.concat([test_csv, one_hot_encoded], axis=1) # 기존 데이터프레임에 원핫인코딩된 결과를 추가합니다.
# test_csv.drop(column_name, axis=1, inplace=True)   # 기존 근로기간 컬럼을 삭제합니다.

# 대출기간 처리 ohe
column_name = '대출기간'
one_hot_encoded = pd.get_dummies(train_csv[column_name], prefix=column_name)
train_csv = pd.concat([train_csv, one_hot_encoded], axis=1)
train_csv.drop(column_name, axis=1, inplace=True)

one_hot_encoded = pd.get_dummies(test_csv[column_name], prefix=column_name)
test_csv = pd.concat([test_csv, one_hot_encoded], axis=1)
test_csv.drop(column_name, axis=1, inplace=True)

# 주택소유상태 ohe
train_csv['주택소유상태'] = train_csv['주택소유상태'].replace({'ANY' : 'MORTGAGE'})

column_name = '주택소유상태'
one_hot_encoded = pd.get_dummies(train_csv[column_name], prefix=column_name)
train_csv = pd.concat([train_csv, one_hot_encoded], axis=1)
train_csv.drop(column_name, axis=1, inplace=True)

one_hot_encoded = pd.get_dummies(test_csv[column_name], prefix=column_name)
test_csv = pd.concat([test_csv, one_hot_encoded], axis=1)
test_csv.drop(column_name, axis=1, inplace=True)


# 대출목적 ohe
test_csv['대출목적'] = test_csv['대출목적'].replace({'결혼' : '부채 통합'})

column_name = '대출목적'
one_hot_encoded = pd.get_dummies(train_csv[column_name], prefix=column_name)
train_csv = pd.concat([train_csv, one_hot_encoded], axis=1)
train_csv.drop(column_name, axis=1, inplace=True)

one_hot_encoded = pd.get_dummies(test_csv[column_name], prefix=column_name)
test_csv = pd.concat([test_csv, one_hot_encoded], axis=1)
test_csv.drop(column_name, axis=1, inplace=True)

# print(train_csv)
num_features = train_csv.shape[1]  
print(num_features)    #  column count train : 28, test : 27


encoder.fit(train_csv['대출등급'])
X = train_csv.drop(['대출등급'], axis=1)    #(96294, 27)
y = train_csv['대출등급']   #(96294,)
#--




y = y.values.reshape(-1, 1)
# y = encoder.transform(train_csv['대출등급'])
y = OneHotEncoder(sparse=False).fit_transform(y)

#--------------------










#-------- sklearn
# y = y.values.reshape(-1, 1)
# y = OneHotEncoder(sparse=False).fit_transform(y)


# model = Sequential()
# model.add(Dense(19, activation='relu', input_shape=(13,)))
# model.add(BatchNormalization())
# model.add(Dense(97, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(9, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(21, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(16, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(21, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(7, activation='softmax'))

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(27,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(7, activation='softmax'))


from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy'
                   , mode='max'
                   , patience=400
                   , verbose=0
                   , restore_best_weights=True
                   )

mms = StandardScaler()

def auto(test_csv) :
    rs = random.randrange(2, 99999999)
    # bs = random.randrange(5000, 11999)
    X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=rs, train_size=0.82, stratify=y)
    
    X_train =np.asarray(X_train).astype(np.float32) 
    X_test =np.asarray(X_test).astype(np.float32)
    test_csv =np.asarray(test_csv).astype(np.float32)
    
    mms.fit(X_train)
    X_train = mms.transform(X_train)
    X_test = mms.transform(X_test)
    test_csv = mms.transform(test_csv)
    
    
    #---mms

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=0, save_best_only=True, filepath=filepath)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, epochs=20000, batch_size=500, validation_split=0.14, callbacks=[es, mcp])
    


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
    return f1, rs


# f1, rs, bs , hist = auto()
# print("f1 : " , f1)

max_f1 = 0.85

while True:
    
    date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
    path = '..\\_data\_save\\MCP\\dacon_loan\\'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
    filepath = ''.join([path, 'loan_test', date, '_' ,filename])
    
    f1, rs = auto(test_csv)
    if f1 > max_f1 :
        max_f1 = f1
        submission_csv.to_csv(csv_path + date + str(rs) + "_f1_" + str(f1) + ".csv", index=False)
        
        if f1 > 0.9:
            break
            
# print(f1)
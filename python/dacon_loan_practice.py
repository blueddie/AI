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

encoder = LabelEncoder()
ohe = OneHotEncoder()

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

# # 근로기간 원핫 인코딩
# column_name = '근로기간'
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

# 대출기간 처리 lae
# encoder.fit(train_csv['대출기간'])
# train_csv['대출기간'] = encoder.transform(train_csv['대출기간'])
# test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])





# 주택소유상태 ohe
train_csv['주택소유상태'] = train_csv['주택소유상태'].replace({'ANY' : 'MORTGAGE'})

# column_name = '주택소유상태'
# one_hot_encoded = pd.get_dummies(train_csv[column_name], prefix=column_name)
# train_csv = pd.concat([train_csv, one_hot_encoded], axis=1)
# train_csv.drop(column_name, axis=1, inplace=True)

# one_hot_encoded = pd.get_dummies(test_csv[column_name], prefix=column_name)
# test_csv = pd.concat([test_csv, one_hot_encoded], axis=1)
# test_csv.drop(column_name, axis=1, inplace=True)

# label
train_csv['주택소유상태'] = train_csv['주택소유상태'].replace({'MORTGAGE' : 3, 'OWN' : 2, 'RENT' : 0})
test_csv['주택소유상태'] = test_csv['주택소유상태'].replace({'MORTGAGE' : 3, 'OWN' : 2, 'RENT' : 0})


encoder.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = encoder.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])


# #---1001
# 대출목적 ohe
test_csv['대출목적'] = test_csv['대출목적'].replace({'결혼' : '부채 통합'})

# column_name = '대출목적'
# one_hot_encoded = pd.get_dummies(train_csv[column_name], prefix=column_name)
# train_csv = pd.concat([train_csv, one_hot_encoded], axis=1)
# train_csv.drop(column_name, axis=1, inplace=True)

# one_hot_encoded = pd.get_dummies(test_csv[column_name], prefix=column_name)
# test_csv = pd.concat([test_csv, one_hot_encoded], axis=1)
# test_csv.drop(column_name, axis=1, inplace=True)

#replcae
train_csv['대출목적'] = train_csv['대출목적'].replace({'부채 통합' : 0, '주택 개선' : 1, '주요 구매' : 2, '휴가' : 3, '의료' : 4, '자동차' : 5, '신용 카드' : 6, '소규모 사업' : 7, '기타' : 8, '이사' : 9 ,'주택': 10,'재생 에너지': 11})
test_csv['대출목적'] = test_csv['대출목적'].replace({'부채 통합' : 0, '주택 개선' : 1, '주요 구매' : 2, '휴가' : 3, '의료' : 4, '자동차' : 5, '신용 카드' : 6, '소규모 사업' : 7, '기타' : 8, '이사' : 9 ,'주택': 10,'재생 에너지': 11})


num_features = test_csv.shape[1]  
print(num_features)    #  column count train : 15, test : 14



# encoder.fit(train_csv['대출등급'])


columns_to_drop = ['대출등급', '최근_2년간_연체_횟수', '연체계좌수', '총연체금액']
X = train_csv.drop(columns=columns_to_drop)
y = train_csv['대출등급']   #(96294,)

test_drop = ['최근_2년간_연체_횟수', '연체계좌수','총연체금액']
test_csv = test_csv.drop(columns=test_drop)
# print(y.shape)    #(96294, 12)

y = y.values.reshape(-1, 1)
# y = encoder.transform(train_csv['대출등급'])
ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y = ohe.transform(y)
# print(y)

print(y.shape)


#--------------
while True:
    rs = random.randrange(2,99999999)
    X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=rs, train_size=0.85, stratify=y)

    # #2 
    # model = Sequential()
    # model.add(Dense(128, activation='swish', input_shape=(27,)))
    # model.add(Dense(50, activation='swish'))
    # model.add(Dense(100, activation='swish'))
    # model.add(Dense(32, activation='swish'))
    # model.add(Dense(16, activation='swish'))
    # model.add(Dense(21, activation='swish'))
    # model.add(Dense(7, activation='softmax'))
    #-------
    model = Sequential()
    model.add(Dense(19, activation='swish', input_shape=(11,)))
    model.add(Dense(97, activation='swish'))
    model.add(Dense(9, activation='swish'))
    model.add(Dense(21, activation='swish'))
    model.add(Dense(23, activation='swish'))
    model.add(Dense(17, activation='swish'))
    model.add(Dense(7, activation='softmax'))

    from keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss'
                    , mode='auto'
                    , patience=1000
                    , verbose=1
                    , restore_best_weights=True
                    )

    date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
    path = '..\\_data\_save\\MCP\\dacon_loan\\'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
    filepath = ''.join([path, 'loan_test', date, '_' ,filename])


    # mcp = ModelCheckpoint(monitor='val_accuracy', mode='auto', verbose=0, save_best_only=True, filepath=filepath)

    X_train =np.asarray(X_train).astype(np.float32) 
    X_test =np.asarray(X_test).astype(np.float32)
    test_csv =np.asarray(test_csv).astype(np.float32)

    print(X_train.shape)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    test_csv = scaler.transform(test_csv)

    # #3
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    hist = model.fit(X_train, y_train, epochs=10, batch_size=600, validation_split=0.2, callbacks=[es])

    #4
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

    submission_csv['대출등급'] = y_submit

    f1 = f1_score(y_test, y_predict, average='macro')

    print("acc : " , acc)
    print('f1 : ', f1)

    file_f1 = str(round(f1, 4))
    
    max_f1 = 0.8
    
    if f1 > max_f1 :
        
        max_f1 = f1
        print(rs)
        model.save('C:\\_data\\_save\\MCP\\dacon_loan\\0122\\best.hdf5')
        
        submission_csv.to_csv(csv_path + date + '_rs_'+ str(rs) +"_f1_" + file_f1 + ".csv", index=False)
        
        if max_f1 >= 0.8 :
            print('random_state : ', rs)
            print('f1 : ' , f1)
            break
        
        

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['val_loss'], color = 'red', label ='val_loss', marker='.')
# plt.plot(hist.history['val_accuracy'], color = 'blue', label ='val_acc', marker='.')
# plt.xlabel = 'epochs'
# plt.ylabel = 'loss'
# plt.show()


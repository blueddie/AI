from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE    # anacon

csv_path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

# df = train_csv

# numeric_columns = df.select_dtypes(include=['int', 'float']).columns

# columns_to_check = numeric_columns[numeric_columns != "대출등급"]   # 대출등급을 제외한 연속형 변수 사용
 
# z_scores = stats.zscore(df[columns_to_check])   # Z-score를 계산하여 이상치 확인   

# threshold = 2   # 임계값 설정 (보통 2 또는 3을 사용)

# outliers = (z_scores > threshold).any(axis=1)   # Z-score가 임계값을 넘어서는 데이터 포인트를 이상치로 간주


# df_no_outliers = df[~outliers]  # 이상치 제거

# print("이상치 제거 전:", df.shape)

# # 이상치 제거 후의 행과 열 수
# print("이상치 제거 후:", df_no_outliers.shape)
# # 이상치 제거 전의 통계량
# print("이상치 제거 전 평균:\n", df[columns_to_check].mean())

# # 이상치 제거 후의 통계량
# print("이상치 제거 후 평균:\n", df_no_outliers[columns_to_check].mean())

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

columns_to_drop = ['대출등급', '최근_2년간_연체_횟수','연체계좌수']
x = xy.drop(columns=columns_to_drop)

columns_to_drop_test = ['최근_2년간_연체_횟수','연체계좌수']
test_csv = test_csv.drop(columns=columns_to_drop_test)
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

# print(x.shape)  #(90293, 13)


#2 모델
model = Sequential()
model.add(Dense(19, activation='swish', input_shape=(11,)))
model.add(Dense(97, activation='swish'))
model.add(Dense(9, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(23, activation='swish'))
model.add(Dense(17, activation='swish'))
model.add(Dense(7, activation='softmax'))

x_train, x_test, y_train, y_test = train_test_split(x, y ,random_state=65923048, train_size=0.84, stratify=y)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
test_csv = test_csv.astype(np.float32)



mms_arr = ['대출금액', '연간소득', '연간소득', '총상환이자', ]

sds_arr = ['대출기간', '근로기간', '주택소유상태','부채_대비_소득_비율', '총계좌수', '대출목적',  ]

rbs_arr = ['최근_2년간_연체_횟수', ]

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=30, verbose=1, restore_best_weights=True)

# date = datetime.datetime.now().strftime("%m%d_%H%M")    

# path = 'c:\\_data\_save\\MCP\\dacon_loan\\'

#3 컴파일, 훈련
from keras.optimizers import Adam
# learning_rate = 1.0
# learning_rate = 0.1
# learning_rate = 0.01
# learning_rate = 0.001
learning_rate = 0.0001

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=77777, batch_size=212, validation_split=0.18, callbacks=[es])
        
#4
results = model.evaluate(x_test, y_test)
loss = results[0]
acc = results[1]

y_predict = model.predict(x_test)
y_predict = ohe.inverse_transform(y_predict)
y_test = ohe.inverse_transform(y_test)

y_submit = model.predict(test_csv)
y_submit = ohe.inverse_transform(y_submit)

y_submit = pd.DataFrame(y_submit)
submission_csv['대출등급'] = y_submit


f1 = f1_score(y_test, y_predict, average='macro')

print("lr : ", learning_rate,'loss : ', loss)
print("lr : ", learning_rate,'acc : ', acc)
print("lr : ", learning_rate,'f1 : ', f1)
# file_f1 = str(round(f1, 5)) 

# model.save('C:\\_data\\_save\\models\\loan\\'+ date + '_f1_'+ file_f1 +'best.hdf5')
# submission_csv.to_csv(csv_path + date + '_f1_' + file_f1 + ".csv", index=False)


# submission_csv['대출등급'] = pd.DataFrame(y_submit.reshape(-1,1))





# lr :  0.1 loss :  1.2610394954681396
# lr :  0.1 acc :  0.44392523169517517
# lr :  0.1 f1 :  0.29052761760162993

# lr :  0.01 loss :  0.3913651406764984
# lr :  0.01 acc :  0.8524143099784851
# lr :  0.01 f1 :  0.8082371002305132

# lr :  0.001 loss :  0.46114927530288696
# lr :  0.001 acc :  0.8386552333831787
# lr :  0.001 f1 :  0.7914621312055916

# lr :  0.0001 loss :  0.49153146147727966
# lr :  0.0001 acc :  0.8333982229232788
# lr :  0.0001 f1 :  0.7799336090649709
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE    # anacon
from keras.callbacks import EarlyStopping



csv_path = "C:\\_data\\dacon\\loan_grade\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

xy = train_csv
##############################################################################################
unknown_replacement = xy['근로기간'].mode()[0]          # 최반값을 담았음
xy.loc[xy['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement       
test_csv.loc[test_csv['근로기간'] == 'Unknown', '근로기간'] = unknown_replacement

# print(np.unique(xy['근로기간'], return_counts=True))
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

encoder = LabelEncoder()
encoder.fit(xy['대출기간'])
xy['대출기간'] = encoder.transform(xy['대출기간'])
test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])

xy.loc[xy['주택소유상태'] == 'ANY', '주택소유상태'] = 'MORTGAGE'

encoder.fit(xy['주택소유상태'])
xy['주택소유상태'] = encoder.transform(xy['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])



test_csv.loc[test_csv['대출목적'] == '결혼', '대출목적'] = '부채 통합'

encoder.fit(xy['대출목적'])
xy['대출목적'] = encoder.transform(xy['대출목적'])
test_csv['대출목적'] = encoder.transform(test_csv['대출목적'])

categorical = ['대출기간', '근로기간', '주택소유상태', '대출목적']
numeric = ['대출금액', '연간소득', '부채_대비_소득_비율', '총계좌수', '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수']

# xy = xy[xy['총상환이자'] != 0.0]

x_categorical = xy[categorical]
x_numeric = xy[numeric]

test_categorical = test_csv[categorical]
test_numeric = test_csv[numeric]

x_categorical = x_categorical.astype(np.float32)
x_numeric = x_numeric.astype(np.float32)

test_categorical = test_categorical.astype(np.float32)
test_numeric = test_numeric.astype(np.float32)


# print(test_categorical.shape)   #(64197, 4)
# print(test_numeric.shape)       #(64197, 9)

y = xy['대출등급']
# print(type(x1))     #<class 'pandas.core.frame.DataFrame'>
# print(type(x2))     #<class 'pandas.core.frame.DataFrame'>
# print(x1.shape) #(96294, 4)
# print(x2.shape) #(96294, 9)
# print(y.shape)  #(96294,)
y = y.values.reshape(-1, 1)
# print(y.shape)  #(96294, 1)
# print(type(y))  #<class 'numpy.ndarray'>
ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y = ohe.transform(y)
# print(y.shape)  #(96294, 7)
# 2-1 모델
input1 = Input(shape=(4,), name='x_cat_train')
dense1 = Dense(19, activation='swish', name='bit1')(input1)
dense2 = Dense(97, activation='swish', name='bit2')(dense1)
dense3 = Dense(9, activation='swish', name='bit3')(dense2)
dense4 = Dense(21, activation='swish', name='bit4')(dense3)
dense5 = Dense(17, activation='swish', name='bit5')(dense4)
output1 = Dense(14, activation='swish', name='bit6')(dense5)


#2-2 모델
input11 = Input(shape=(9,), name='x_num_train')
dense11 = Dense(19, activation='swish', name='bit11')(input11)
dense12 = Dense(97, activation='swish', name='bit12')(dense11)
dense13 = Dense(9, activation='swish', name='bit13')(dense12)
dense14 = Dense(21, activation='swish', name='bit14')(dense13)
dense15 = Dense(17, activation='swish', name='bit15')(dense14)
output11 = Dense(14, activation='swish', name='bit16')(dense15)

# 2-3 concatenate
merge1 = concatenate([output1, output11], name='mg1')
merge2 = Dense(19, activation='swish', name='mg2')(merge1)
merge3 = Dense(97, activation='swish', name='mg3')(merge2)
merge4 = Dense(9, activation='swish', name='mg4')(merge3)
merge5 = Dense(21, activation='swish', name='mg5')(merge4)
merge6 = Dense(23, activation='swish', name='mg6')(merge5)
merge7 = Dense(17, activation='swish', name='mg7')(merge6)
last_output = Dense(7,activation='softmax', name='last')(merge7)

model = Model(inputs=[input1, input11], outputs=last_output)
model.summary()
#3

x_cat_train, x_cat_test, x_num_train, x_num_test, y_train, y_test = train_test_split(x_categorical, x_numeric, y, random_state=3777, train_size=0.9, stratify=y)

# print(x_cat_train.shape)
sds = StandardScaler()
sds.fit(x_cat_train)
x_cat_train = sds.transform(x_cat_train)
x_cat_test = sds.transform(x_cat_test)
test_categorical = sds.transform(test_categorical)

rbs = RobustScaler()
rbs.fit(x_num_train)
x_num_train = rbs.transform(x_num_train)
x_num_test = rbs.transform(x_num_test)
test_numeric = rbs.transform(test_numeric)

es = EarlyStopping(monitor='val_loss', mode='min', patience=200, verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit([x_cat_train, x_num_train], y_train, batch_size=279, validation_split=0.1, epochs=10000, callbacks=[es])

#4
results = model.evaluate([x_cat_test, x_num_test], y_test)
loss = results[0]
acc = results[1]

y_predict = model.predict([x_cat_test, x_num_test])
y_predict = ohe.inverse_transform(y_predict)
y_test = ohe.inverse_transform(y_test)

y_submit = model.predict([test_categorical, test_numeric])
y_submit = ohe.inverse_transform(y_submit)
y_submit = pd.DataFrame(y_submit)

submission_csv['대출등급'] = y_submit

f1 = f1_score(y_test, y_predict, average='macro')

print('f1 스코어 : ', f1)
file_f1 = str(round(f1, 4))
date = datetime.datetime.now().strftime("%m%d_%H%M")    
submission_csv.to_csv(csv_path + date +'_f1_'+ file_f1 +".csv", index=False)


#f1 스코어 :  0.8555154190928889

# f1 스코어 :  0.9000598098071749   bs 256 rs 3 ts 0.84

# f1 스코어 :  0.8890332639190869   256/13/0.84


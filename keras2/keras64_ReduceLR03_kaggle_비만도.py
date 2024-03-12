from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, RandomizedSearchCV, GridSearchCV
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import warnings
from scipy.stats import uniform, randint
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")
def outlierHandler(data, labels):
    data = pd.DataFrame(data)
    
    for label in labels:
        series = data[label]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        
        # print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
        data = data.fillna(data.ffill())
        data = data.fillna(data.bfill())

    return data

#1. 데이터
csv_path = 'C:\\_data\\kaggle\\obesity\\'

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

xy = train_csv.copy()
x_pred = test_csv.copy()

columns_to_drop = ['NObeyesdad']
x = xy.drop(columns=columns_to_drop)
y = xy[columns_to_drop]

non_float_x = []
numeric_x = []
for col in x.columns:
    if x[col].dtype != 'float64':
        non_float_x.append(col)
    else :
        numeric_x.append(col)
        
# print(numeric_x)
x = outlierHandler(x, numeric_x)
# print(pd.isna(x).sum())
# print(non_float)    #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
non_float_pred = []
numeric_test = []
for col in x_pred.columns:
    if x_pred[col].dtype != 'float64':
        non_float_pred.append(col)
    else :
        numeric_test.append(col)
# print(non_float_pred)   #['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
print(numeric_test)
x_pred = outlierHandler(x_pred, numeric_test)


for col in non_float_x:
    print(f'x : {pd.value_counts(x[col])}')
    print(f'x_pred : {pd.value_counts(x_pred[col])}')
    print('------------------------------------')

# CALC -> Always 2 train에 없는 라벨 있음
x_pred['CALC'] = x_pred['CALC'].replace({'Always' : 'Sometimes'})
# print(pd.value_counts(x_pred['CALC']))

for column in x.columns:
    if (x[column].dtype != 'float64'):
        encoder = LabelEncoder()
        x[column] = encoder.fit_transform(x[column])
        x_pred[column] = encoder.transform(x_pred[column])
            
for col in x.columns :
    if x[col].dtype != 'float32':
        x[col] = x[col].astype('float32')
        x_pred[col] = x_pred[col].astype('float32')
# print(x.dtypes)
# print(x_pred.dtypes)


ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)
# print(len(y[0]))
rs = 10005
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, random_state=rs, test_size=0.2)

# print(x_train.shape)    #(16606, 16)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_pred = scaler.transform(x_pred)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
# test_csv = test_csv.astype(np.float32)

model = Sequential()
model.add(Dense(32, activation='swish' ,input_shape=(16, )))
model.add(Dense(64, activation='swish'))
model.add(Dense(32, activation='swish'))
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(32, activation='swish'))
model.add(Dense(8, activation='swish'))
model.add(Dense(7, activation='softmax'))
model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=30, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, period=5, filepath="")
rlr = ReduceLROnPlateau(monitor='val_loss',             # 통상 early_stopping patience보다 작다
                        patience=10,
                        mode='min',
                        verbose=1,
                        factor=0.5,
                        # 통상 디폴트보다 높게 잡는다?
                        )
#3 컴파일, 훈련
from keras.optimizers import Adam
learning_rate = 0.01


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=77777, batch_size=212, validation_split=0.18, callbacks=[es, mcp, rlr])

#4
results = model.evaluate(x_test, y_test)
loss = results[0]
acc = results[1]

y_predict = model.predict(x_test)
y_predict = ohe.inverse_transform(y_predict)
y_test = ohe.inverse_transform(y_test)

# y_submit = model.predict(test_csv)
# y_submit = ohe.inverse_transform(y_submit)

print("lr : ", learning_rate,'loss : ', loss)
print("lr : ", learning_rate,'acc : ', acc)

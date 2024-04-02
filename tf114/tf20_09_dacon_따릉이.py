import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

#1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv")   #, index_col=0

xy = train_csv.copy()

x = xy.drop(['count'], axis=1)
y = xy['count']

# print(x.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5'],
#       dtype='object')

# print(pd.isna(x).sum())
# hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117

# print(pd.isna(test_csv).sum())
# hour                       0
# hour_bef_temperature       1
# hour_bef_precipitation     1
# hour_bef_windspeed         1
# hour_bef_humidity          1
# hour_bef_visibility        1
# hour_bef_ozone            35
# hour_bef_pm10             37
# hour_bef_pm2.5            36

x = x.astype('float32')
test_csv = test_csv.astype('float32')
# 결측치 처리
x = x.interpolate() 
test_csv = test_csv.interpolate()

# 결측치 없음을 확인
# print(pd.isna(test_csv).sum())
# print(pd.isna(x).sum())
# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0

def outlierHandler(data):
    data = pd.DataFrame(data)
    
    for label in data:
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

x = outlierHandler(x)
test_csv = outlierHandler(test_csv)
# 이상치 nan 처리 후 interpolate
# print(pd.isna(x).sum())
# print(pd.isna(test_csv).sum())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1226)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape, y_train.shape) #(1313, 9) (1313,)
print(x_test.shape, y_test.shape) #(146, 9) (146,)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 9])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None,])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9,1]), tf.float32, name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), tf.float32, name='bias')

# 2. 모델
hypothesis = tf.matmul(xp, w) + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - yp)) # mse
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101


    
for step in range(epochs):
    _, loss_val, w_v, b_v = sess.run([train, loss, w, b], feed_dict={xp:x_train, yp:y_train})

    
    print("epochs:", step + 1, "\t", loss_val)


x_pred = tf.compat.v1.placeholder(tf.float32, shape=[None, 9])
y_pred = tf.matmul(x_pred, w_v) + b_v

y_pred = sess.run(y_pred, feed_dict={x_pred: x_test})

r2 = r2_score(y_test, y_pred)
print("r2 score : ", r2)
sess.close()
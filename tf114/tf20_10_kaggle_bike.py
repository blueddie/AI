# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import random
import time, datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
#1. 데이터
csv_path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sampleSubmission.csv")

x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

x = x.astype('float32')
test_csv = test_csv.astype('float32')

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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=15)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# print(test_csv.shape)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# print(x_train.shape, y_train.shape) #(8708, 8) (8708,)
# print(x_test.shape, y_test.shape)   #(2178, 8) (2178,)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None,])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1]), tf.float32, name='weight')
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


x_pred = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y_pred = tf.matmul(x_pred, w_v) + b_v

y_pred = sess.run(y_pred, feed_dict={x_pred: x_test})

r2 = r2_score(y_test, y_pred)
print("r2 score : ", r2)
sess.close()
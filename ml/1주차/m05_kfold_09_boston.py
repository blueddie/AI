from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
import datetime
from sklearn.svm import LinearSVR

datasets = load_boston()

x = datasets.data
y = datasets.target
# print(x.shape)  #(506, 13)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8)


#2
model = LinearSVR(C=24)

#3. 컴파일 훈련
model.fit(x_train, y_train)
results = model.score(x_test, y_test)
print('r2 score : :' , results)
# R2 score :  0.7615445774567593
# loss :  3.145082473754883

# R2 score :  0.6817796619773973
# loss :  3.125746965408325

# R2 score :  0.6736491113441989
# loss :  3.5278685092926025

# R2 score :  0.7359853474011155
# loss :  2.8940417766571045

# r2 score : : 0.6733497735858817


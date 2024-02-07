# https://dacon.io/competitions/open/236070/overview/description
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder

#1.
path = "C://_data//dacon//iris//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

x = train_csv.drop(['species'], axis=1)
y = train_csv['species']

# print(X.shape)  #(120, 4)
# print(y.shape)  #(120,)
print(pd.value_counts(y))



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=13, train_size=0.8)
#2
#3


# model.fit(X_train, y_train, epochs=200, batch_size=1
#         , validation_split=0.2
#         , callbacks=[es]
#         )




    


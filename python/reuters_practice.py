from keras.datasets import reuters
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.utils import pad_sequences
from keras.layers import Dense, Conv1D, Flatten, LSTM, Embedding
from keras.models import Sequential

#1
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000
                                                         , test_split=0.2)

print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,)

print(type(x_train[0])) #<class 'list'>
print(np.unique(y_train, return_counts=True))

len_list = [len(i) for i in x_train] + [len(i) for i in x_test]
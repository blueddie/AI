from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

text = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}

print(token.word_counts)
#OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print(x)
# [[4, 2, 2, 3, 3, 5, 6, 7, 1, 1, 1, 8]]

from keras.utils import to_categorical

#1. to categorical에서 첫번째 0 빼기
x_array = np.array(x)
# x1 = to_categorical(x)
# print(x1)            # 3차원이다? RNN
# print(x1.shape)      # (1, 12, 9)
# [[[0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]

# print(x1.shape)   #(1, 12, 9)


#2 사이킷런 원핫
x = np.array(x).reshape(-1,1)
print(x.shape)
ohe = OneHotEncoder(sparse=False)
x2 = ohe.fit_transform(x)
print(x2)
print(x2.shape) #(12, 8)


# #3 판다스 겟더미
# x = np.array(x).reshape(-1,)
# x3 = pd.get_dummies(x, dtype=int)
# print(x3)
# print(x3.shape) #(12, 8)
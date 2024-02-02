from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import pad_sequences
import pandas as pd
import numpy as np

text1 = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '상헌이가 선생을 괴롭힌다. 상헌이는 못생겼다. 상헌이는 마구 마구 못생겼다.'

token =Tokenizer()
token.fit_on_texts([text1,text2])              # 2개 이상도 가능

# print(token.word_index)

{'마구': 1, '진짜': 2, '매우': 3, '상헌이는': 4, '못생겼다': 5, '나는': 6, '맛있는': 7, '밥을': 8, '엄청': 9, '먹었다': 10, '상헌이가': 11, '선생을': 12, '괴롭힌다': 13}

# print(token.word_counts)
# OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 5), ('먹었다', 1), ('상헌이가', 1), ('선생을', 1), ('괴롭힌다', 1), ('상헌이는', 2), ('못생겼다', 2)])
x = token.texts_to_sequences([text1, text2])
x_padded = pad_sequences(x, padding='pre',maxlen=12)

print(x)
# [[6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 10], [11, 12, 13, 4, 5, 4, 1, 1, 5]]
print(x_padded)
# [[ 6  2  2  3  3  7  8  9  1  1  1 10]
#  [ 0  0  0 11 12 13  4  5  4  1  1  5]

# 1. to_categorical
# x1 = to_categorical(x_padded)
# print(x1.shape) #(2, 12, 14)
# x1 = x1[:,:-3,1:]
# print(x1)
# print(x1)

#2. ohe
x = np.array(x_padded)
print(x.shape)  #(2, 12)    
print(x_padded.shape)   #(2, 12)
x_padded = x.reshape(-1,1)
print(x_padded.shape)   #(24, 1)
ohe = OneHotEncoder(sparse=False)
x2 = ohe.fit_transform(x_padded)
print(x2.shape) #(24, 14)




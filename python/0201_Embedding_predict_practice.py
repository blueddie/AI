from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv1D, Flatten, LSTM, Embedding
from keras.models import Sequential


#1 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요'
    , '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄'
    , '별로에요', '생각보다 지루해요', '연기가 어색해요'
    , '재미없어요.', '너무 재미없다', '참 재밋네요'
    , '상헌이 바보', '반장 잘생겼다.', '욱이 또 잔다'
]

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,])

x_predict = '나는 정룡이가 정말 싫다. 재미없다 너무 정말'

all_text = docs.copy()
all_text.append(x_predict)
print(all_text)

token = Tokenizer()
token.fit_on_texts(all_text)
print(token.word_index)

word_size = len(token.word_index) + 1
print(word_size)    # 35

all_text_list = token.texts_to_sequences(all_text)
print(all_text_list)

all_text_list = pad_sequences(all_text_list)
print(all_text_list)

x = all_text_list[:len(docs)]
x_predict = all_text_list[len(docs):]

print(x.shape)          #(15, 7)
print(x_predict.shape)  #(1, 7)

#2
model = Sequential()
model.add(Embedding(word_size, 10))#, input_length=5))       # 단어 사전의 개수, 출력 노드의 수, 단어의 길이
model.add(LSTM(512, input_shape=(5,1), activation='relu')) 
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))


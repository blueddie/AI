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

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, 
#  '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, 
#  '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '상헌이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '욱이': 28, '또': 29, '잔다': 30}
x = token.texts_to_sequences(docs)
print(x)
'''
# [[2, 3], [1, 4], [1, 5, 6]
# , [7, 8, 9], [10, 11, 12, 13, 14]  [15],
# [16], [17, 18][19, 20], 
# [21], [2, 22] [1, 23],
# [24, 25] [26, 27], [28, 29, 30]]
print(type(x))  #<class 'list'>
# x = np.array(x) # 길이가 맞지 않아서 패딩을 준다. 시계열 데이터이기 때문에 뒤에 보다는 앞에 주는 게 좋다. 정답은 없다.
pad_x = pad_sequences(x
                      , padding='pre'       # default='pre'
                      , maxlen=5
                      , truncating='pre'    # default='pre'
                      )
print(pad_x)

print(pad_x.shape)   #(15, 5)

# pad_x = pad_x.reshape(-1, 5, 1) #2차원을 그냥 LSTM에 넣으면 알아서 돌아?


#2. 모델
model = Sequential()
########################임베딩1
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5))   # 단어 사전의 개수, 출력 노드의 수, 단어의 길이
# 임베딩 연산량 =  input_dim * output_dim = 31 * 100 = 3100
# 임베딩 인풋의 shape : 2차원 , 임베딩의 아웃풋 shape : 3차원   보통 자연어 처리는 임베딩 후 LSTM이 쌍으로 따라간다.
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100
#  lstm (LSTM)                 (None, 10)                4440
#  dense (Dense)               (None, 7)                 77
#  dense_1 (Dense)             (None, 1)                 8
# =================================================================
# Total params: 7,625
# Trainable params: 7,625
# Non-trainable params: 0

######################2 임베딩2
# model.add(Embedding(input_dim=31, output_dim=100)) 

# #  Layer (type)                Output Shape              Param #
# #  embedding (Embedding)       (None, None, 100)         3100
# #  lstm (LSTM)                 (None, 10)                4440
# #  dense (Dense)               (None, 7)                 77
# #  dense_1 (Dense)             (None, 1)                 8

# # Total params: 7,625
# # Trainable params: 7,625
# # Non-trainable params: 0

# model.add(Embedding(input_dim=31, output_dim=100))
# # input_dim=31 # 디폴트
# # input_dim=29 # 단어 사전의 갯수보다 작을 때 .. 연샨량이 줄어, 단어사전에서 임의로 뺴 : 성능 저하 가능성이 있음
# # input_dim=40 # 단어 사전의 갯수보다 클 때  .. 연산량이 늘어, 임의의 랜덤 임베딩 생성 : 성능 저하 가능성이 있음

#################################임베딩 3
# model.add(Embedding(31, 100))   # 잘 동아감
# model.add(Embedding(31, 100, 5))   # 에러 
model.add(Embedding(31, 100, input_length=5))   # 
# input_length 1,5 는 돼, 2,3,4,6 안 돼 모르면 그냥 None 해
# 임베딩 레이어는 벡터화 시킨 다음 3차원 데이터로 출력해준다.
model.add(LSTM(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()



#3 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(pad_x, labels, epochs=500, batch_size=15)

#4
results = model.evaluate(pad_x, labels)
print(model.predict(pad_x))
print(results)


# [5.349873026716523e-05, 1.0]
x_prdict = '나는 정룡이가 정말 싫다. 재미없다 너무 정말'
x_predict = x_prdict.split(' ')

token.fit_on_texts(x_prdict)
x_predict = token.texts_to_sequences(x_prdict)
print(x_predict)

print(token.word_index)
x_preidct = np.array(x_preidct)

x_preidct = x_preidct.reshape(-1, 1)
print(x_preidct.shape)

pad_pre_x = pad_sequences(x_preidct
                      , padding='pre'       # default='pre'
                      , maxlen=5
                      , truncating='pre'    # default='pre'
                      )


y_predict = model.predict(pad_pre_x)

# {'너무': 1, '참': 2, '재미없다': 3, '정말': 4, '재미있다': 5, '최고에요': 6, '잘만든': 7, '영화예요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, '한': 12, '번': 13, '더': 14, '보고': 15, '싶어요': 16, '글쎄': 17, '별로에요': 18
#  , '생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22, '재미없어요': 23, '재밋네요': 24, '상헌이': 25, '바보': 26, '반장': 27, '잘생겼다': 28, '욱이': 29, '또': 30, '잔다': 31, '나는': 32, '정룡이가': 33, '싫다': 34}


# x_preidct = token.texts_to_sequences(x_preidct)
# print(x_preidct)
# 결과는? 긍정? 부정?


'''
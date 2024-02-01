from keras.preprocessing.text import Tokenizer

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

x1 = to_categorical(x)
print(x1)
print(x1.shape)
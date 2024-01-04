import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

#[실습] 넘파이 리스트의 슬라이싱 !! 7:3으로 자르기
x_train = x[:7]     # == [:-3]    #[1 2 3 4 5 6 7]
y_train = y[:7]                   #[1 2 3 4 6 5 7]

x_test = x[7:]  # == [-3:]      #[ 8  9 10]
y_test = y[7:]      #[ 8  9 10]


'''
    위처럼 데이터를 분할했을 때 문제점이 있다. 
    1-7까지 학습하고 8, 9, 10을 테스트하는 건 범위를 벗어나게 된다
    범위 내 데이터에서 일부를 추출한다. ex) 3, 7, 9 
    30%를 추출한다면 범위 안에서 랜덤으로 뽑는다
    
'''

print(x_train)
print(y_train)
print(x_test)
print(y_test)


# #2. 
model = Sequential()
model.add(Dense(1, input_dim=1))

#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3100, batch_size=3)

#4.
loss = model.evaluate(x_test, y_test)
results = model.predict([11000, 7])
print("loss : ", loss)
print("예측값은 : ", results)


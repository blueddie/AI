import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. 데이터 
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([1,2,3,4,5,6,7])
y_test = np.array([1,2,3,4,5,6,7])

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)

x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape)    # (7,1)
print(x_train, y_train) #tensor([1., 2., 3.]) tensor([1., 2., 3.])

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
# model = nn.Linear(1, 1).to(DEVICE) #output, input
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 12),
    nn.Linear(12, 8),
    nn.Linear(8, 1),
).to(DEVICE)

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()                #criterion : 표준
optimizer = optim.Adam(model.parameters(), lr = 0.1)
# optimizer = optim.SGD(model.parameters(), lr = 0.1)

# model.fit(x,y, epochs = 100, batch_size=1)
def train(model, criterion, optimizer, x_train, y_train):
    # model.train()   #훈련모드 default 

    optimizer.zero_grad()
    # w = w - lr * (loss를 weight로 미분한 값)
    hypothesis = model(x_train) #예상치 값 (순전파)
    loss = criterion(hypothesis, y_train) #예상값과 실제값 loss

    #역전파
    loss.backward() #기울기(gradient) 계산 (loss를 weight로 미분한 값)
    optimizer.step() # 가중치(w) 수정(weight 갱신)

    return loss.item() #item 하면 numpy 데이터로 나옴


epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch {}, loss: {}'.format(epoch, loss)) #verbose

print("===================================")

#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model, criterion, x_test, y_test):
    model.eval() #평가모드 드롭아웃이나, 배치노말 같은 것들 비활성화.

    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_test, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", loss2)

#result = model.predict([4])
result = model(torch.Tensor([[11]]).to(DEVICE))
print('11의 예측값 : ', result.item())



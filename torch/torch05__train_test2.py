import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

# train_test_split을 사용하여 데이터를 훈련 세트와 테스트 세트로 분리합니다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# numpy 배열을 PyTorch 텐서로 변환하고, GPU가 사용 가능하면 GPU로 이동시킵니다.
x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, x_test.shape)  # 예: (80, 1)
print(y_train.shape, y_train.shape)
print(x_train, y_train)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 훈련 함수 정의
def train(model, criterion, optimizer, x_train, y_train):
    model.train()  # 훈련 모드

    optimizer.zero_grad()
    hypothesis = model(x_train)  # 순전파: 예측값 계산
    loss = criterion(hypothesis, y_train)  # 손실 계산

    # 역전파
    loss.backward()  # 기울기 계산
    optimizer.step()  # 가중치 갱신

    return loss.item()

# 평가 함수 정의
def evaluate(model, criterion, x_test, y_test):
    model.eval()  # 평가 모드
    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_test, y_predict)
    return loss2.item()

# 훈련 루프
epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 10 == 0 or epoch == epochs:
        val_loss = evaluate(model, criterion, x_test, y_test)
        print(f'epoch {epoch}, loss: {loss}, val_loss: {val_loss}')
    else:
        print(f'epoch {epoch}, loss: {loss}')

print("===================================")

# 최종 평가
loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", loss2)

# 예측
result = model(torch.Tensor([[101]]).to(DEVICE))
print('11의 예측값 : ', result.item())

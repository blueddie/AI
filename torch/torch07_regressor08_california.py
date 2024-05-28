# 07. load_didabetes
# 08. california
# 09. dacon_따릉이
# 10. kaggle_bike

#평가는 rmse, r2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# DEVICE = 'cpu'
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(20640, 8) (20640,)
# train_test_split을 사용하여 데이터를 훈련 세트와 테스트 세트로 분리합니다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape, x_test.shape)

print(y_train.shape, y_test.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)


# numpy 배열을 PyTorch 텐서로 변환하고, GPU가 사용 가능하면 GPU로 이동시킵니다.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 7),
    nn.ReLU(),
    nn.Linear(7, 1),
    # nn.Sigmoid()
).to(DEVICE)

# 3. 컴파일, 훈련
# criterion = nn.MSELoss()
# 사용자 정의 RMSE 손실 함수 정의
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))
    


criterion = RMSELoss()

# optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

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
from sklearn.metrics import accuracy_score

loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", loss2)

# y_pred = np.round(model(x_test).cpu().detach().numpy())
y_pred = model(x_test).cpu().detach().numpy()
y_test = y_test.cpu().numpy()
# .cpu()안했을 때  device='cuda:0', grad_fn=<SigmoidBackward0>
print(y_pred)
print(y_test)

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)

# 예측 함수 정의
# def predict(model, x):
#     model.eval()  # 평가 모드
#     with torch.no_grad():
#         predictions = model(x)
#         predicted_classes = (predictions >= 0.5).float()  # 0.5를 기준으로 이진값으로 변환
#     return predicted_classes

# # 예측
# predictions = predict(model, x_test)
# print('x_test의 예측값 : ', predictions)

'''
rmse :  0.56271994
'''
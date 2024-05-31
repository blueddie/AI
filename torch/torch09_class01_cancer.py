import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# DEVICE = 'cpu'
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)

# train_test_split을 사용하여 데이터를 훈련 세트와 테스트 세트로 분리합니다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print(x_train.shape, x_test.shape)

print(y_train.shape, y_test.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

# bce는 입력과 타겟의 텐서가 같아야 함. 아래 코드처럼 LongTensor 하면 에러
# y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)


# numpy 배열을 PyTorch 텐서로 변환하고, GPU가 사용 가능하면 GPU로 이동시킵니다.

print(x_train.shape)  # 예: (80, 1)
print(x_train, y_train)

# 2. 모델구성
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.Linear(16, 7),
#     nn.ReLU(),
#     nn.Linear(7, 1),
#     nn.Sigmoid()
# ).to(DEVICE)

class Model(nn.Module) :
    def __init__(self, input_dim, output_dim):
        # super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    # 순전파
    def forward(self, input_size):

        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        
        return x

model = Model(30, 1).to(DEVICE)

# 3. 컴파일, 훈련
# criterion = nn.MSELoss()
criterion = nn.BCELoss()

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
        loss2 = criterion(y_predict, y_test)
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


y_pred = np.round(model(x_test).cpu().detach().numpy())


y_test = y_test.cpu().numpy()
# .cpu()안했을 때  device='cuda:0', grad_fn=<SigmoidBackward0>
print(y_pred)
print(y_test)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)
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


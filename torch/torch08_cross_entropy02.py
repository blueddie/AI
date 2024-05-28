# 02. digits
# 03. fetch_covtype
# 04. dacon_wine
# 05. dacon_대출
# 06. kaggle_비만도

# critetion = nn.CrossEntropyLoss()

# torch.argmax()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (1797, 64) (1797,)
# train_test_split을 사용하여 데이터를 훈련 세트와 테스트 세트로 분리합니다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)  # CrossEntropyLoss는 LongTensor를 필요로 함
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)  # 출력 크기를 클래스 수인 10으로 설정
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 훈련 함수 정의
def train(model, criterion, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

# 평가 함수 정의
def evaluate(model, criterion, x_test, y_test):
    model.eval()
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
loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", loss2)

# 예측값 계산 및 평가
y_pred = model(x_test).cpu().detach().numpy()
y_test_np = y_test.cpu().numpy()

# y_pred에서 가장 높은 값의 인덱스를 예측 클래스라고 간주
y_pred_classes = np.argmax(y_pred, axis=1)

# 정확도 계산
accuracy = accuracy_score(y_test_np, y_pred_classes)
print("Accuracy: ", accuracy)


# 예측
print(y_test.shape, y_pred_classes.shape)

f1 = f1_score(y_test_np, y_pred_classes, average='macro')

print("f1 score : " , f1)

'''
최종 loss :  0.32918092608451843
f1 score :  0.9576273523681754
Accuracy:  0.9583333333333334
'''

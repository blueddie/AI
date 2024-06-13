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
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target - 1

print(x.shape, y.shape)  # (581012, 54) (581012,)
print(np.unique(y))


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

# DataLoader를 사용하여 데이터 배치 크기를 조정합니다.
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(54, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 7)  # 
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 훈련 함수 정의
def train(model, criterion, optimizer, train_loader):
    # model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 평가 함수 정의
def evaluate(model, criterion, test_loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_predict = model(x_batch)
            loss = criterion(y_predict, y_batch)
            total_loss += loss.item()
            all_preds.append(y_predict.cpu())
            all_labels.append(y_batch.cpu())
    return total_loss / len(test_loader), torch.cat(all_preds), torch.cat(all_labels)

# 훈련 루프
epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch % 10 == 0 or epoch == epochs:
        val_loss = evaluate(model, criterion, test_loader)
        print(f'epoch {epoch}, loss: {loss}, val_loss: {val_loss}')
    else:
        print(f'epoch {epoch}, loss: {loss}')

print("===================================")

# 최종 평가
loss2 = evaluate(model, criterion, test_loader)
print("최종 loss : ", loss2)

# 예측값 계산 및 평가
y_pred = model(x_test).cpu().detach().numpy()
y_test_np = y_test.cpu().numpy()

# y_pred에서 가장 높은 값의 인덱스를 예측 클래스라고 간주
y_pred_classes = np.argmax(y_pred, axis=1) + 1

# 정확도 계산
accuracy = accuracy_score(y_test_np, y_pred_classes)
print("Accuracy: ", accuracy)


# 예측
print(y_test.shape, y_pred_classes.shape)

f1 = f1_score(y_test_np, y_pred_classes, average='macro')

print("f1 score : " , f1)

'''
Accuracy:  0.8631532748724215
torch.Size([116203]) (116203,)
f1 score :  0.6992999270639909
'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=123, train_size=0.8, stratify=y)

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = SVC(gamma=gamma, C=C)
        model.fit(x_train, y_train)
        
        score = model.score(x_test, y_test)
        
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma' : gamma}
            a = 1

print(f'best score : {best_score:.2f}\nbest parameters : {best_parameters}')
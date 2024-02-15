from xgboost import XGBClassifier
from keras.datasets import mnist
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')
#1 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# print(x.shape, y.shape) #(70000, 28, 28) (70000,)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
y = y.reshape(-1, 1)
# print(y.shape) #(70000, 1)


parameters = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 6, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

n_components = [154, 331, 486, 713, 784]

n_splits = 3
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=8663)

final_results = []

for n_component in n_components:
    pca = PCA(n_components=n_component)
    x_pca = pca.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, train_size=0.8, random_state=77)
    
    model = RandomizedSearchCV(XGBClassifier(tree_method='hist', device='cuda')
                               , parameters
                               , cv=kfold
                               , verbose=1
                               , n_iter=10
                               , n_jobs=-2
                               , random_state=42
                               )

    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    
    # print('best_score : ', model.best_score_)
    # print('model.score : ', model.score(x_test, y_test))
    final_results.append({"n_component": n_component, "best_score" : model.best_score_, "model.score" : model.score(x_test, y_test), "time" : round(end_time - start_time , 3)})

for result in final_results:
    print(result)

# xgboost와 그리드 서치 , 랜덤서치, Halving 등 사용
# njobs = -1

# tree_method='gpu'
# predictor = 'gpu_predictor'
# gpu_id=0
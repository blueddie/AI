from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
#1
datasets = fetch_covtype()

X = datasets.data
y = datasets.target

# DataFrame
columns = datasets.feature_names
X = pd.DataFrame(X, columns=columns)
y = pd.Series(y)
print(X.shape)
# 하위 20퍼센트의 인덱스: [ 2  8 12 14 18 25] 
to_delete = [2, 8, 12, 14, 18]
for idx in sorted(to_delete, reverse=True):
    x = x.drop(x.columns[idx], axis=1)
print(x.shape)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=15152, train_size=0.8)

#2
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models :
    
    #3 
    model.fit(x_train, y_train)

    #4
    results = model.score(x_test, y_test)
    print(model.__class__.__name__, "acc :", results)  

    print(f"{model.__class__.__name__} feature importance\n{model.feature_importances_}")
    
    threshold = np.percentile(model.feature_importances_, 20)  
    low_importance_indices = np.where(model.feature_importances_ < threshold)[0]   
    print("하위 20퍼센트의 인덱스:", low_importance_indices)
    
    
# DecisionTreeClassifier acc : 0.9391237747734568
# DecisionTreeClassifier feature importance
# [3.35864752e-01 2.85063910e-02 1.69748710e-02 6.37000698e-02
#  4.42461483e-02 1.49498330e-01 2.99261546e-02 3.25157271e-02
#  2.35618045e-02 1.41873483e-01 8.14171185e-03 3.71583560e-03
#  1.30057618e-02 2.18805520e-03 3.37951090e-04 9.89756374e-03
#  1.73627055e-03 1.18171624e-02 5.88070141e-04 3.91671231e-04
#  0.00000000e+00 2.17400108e-05 1.03043531e-04 3.44453150e-03
#  2.07883668e-03 9.12964173e-04 2.48061960e-03 1.72197157e-04
#  1.26474905e-05 9.68455537e-04 1.45864476e-03 5.70209998e-06
#  8.79451409e-04 2.82644021e-03 6.50937422e-04 8.25387646e-03
#  9.85211679e-03 4.86392742e-03 7.53945793e-05 9.81547214e-05
#  5.83393750e-04 1.15178829e-04 7.19655923e-03 2.82605149e-03
#  5.11446740e-03 1.22241209e-02 5.29117745e-03 3.43902430e-04
#  8.97730968e-04 1.17556203e-04 1.63736942e-04 2.27682579e-03
#  3.46093389e-03 1.74089808e-03]
# RandomForestClassifier acc : 0.9552335137647049
# RandomForestClassifier feature importance
# [2.50702807e-01 4.73905866e-02 3.23342741e-02 6.08275309e-02
#  5.74725429e-02 1.17939006e-01 4.06626939e-02 4.26976702e-02
#  4.10742344e-02 1.11314717e-01 1.16882531e-02 5.51562983e-03
#  1.15873712e-02 2.94699433e-02 1.05086412e-03 9.60845603e-03
#  2.26009567e-03 1.18144971e-02 5.54896089e-04 2.34130832e-03
#  1.04156971e-05 4.76866610e-05 9.65244191e-05 1.08021810e-02
#  2.58211062e-03 7.85433310e-03 3.74186387e-03 3.82805398e-04
#  5.44414627e-06 7.85418059e-04 1.73927834e-03 2.02547394e-04
#  9.28682225e-04 1.88028926e-03 7.03470655e-04 1.43257632e-02
#  1.01826814e-02 3.88401616e-03 1.44509404e-04 3.89124521e-04
#  6.64774210e-04 1.47455825e-04 5.60749031e-03 3.06272063e-03
#  3.82644102e-03 5.83559172e-03 4.48287508e-03 5.47939865e-04
#  1.39637519e-03 7.78902498e-05 5.45111399e-04 9.64493276e-03
#  9.79402267e-03 5.36985675e-03]

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

x = np.arange(8).reshape(4, 2)
print(x)
# [[0 1]    ->  0,  0,  1   -> y값
#  [2 3]    ->  4,  6,  9
#  [4 5]    -> 16, 20, 25
#  [6 7]]   -> 36, 42, 49

pf = PolynomialFeatures(degree=2    # degree= 차원 수? y값의 형태가 다항식이라고 판단할 때
                        , include_bias=False    # 디폴트 True
                        )   
x_pf = pf.fit_transform(x)
# print(x_pf) # include_bias=True
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]
# print(x_pf) # include_bias=False
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]
pf = PolynomialFeatures(degree=3, include_bias=False) # degree=3 까지 쓰는 경우는 거의 없었다. 해보면 알겠지 뭐
# x_pf = pf.fit_transform(x)
# print(x_pf)

print("========================= 컬럼 3개 ==============================")
x = np.arange(12).reshape(4, 3)
print(x)
pf = PolynomialFeatures(degree=2    # degree= 차원 수? y값의 형태가 다항식이라고 판단할 때
                        , include_bias=False    # 디폴트 True
                        )   
x_pf = pf.fit_transform(x)
print(x_pf)
# [[  0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  9.  10.  11.  81.  90.  99. 100. 110. 121.]]


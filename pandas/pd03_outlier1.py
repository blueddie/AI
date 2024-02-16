import numpy as np

aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    
    return np.where((data_out>upper_bound) |
                    (data_out<lower_bound))
    
outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

# 주어진 데이터에서 사분위수(quantile)를 계산합니다. 이를 위해 np.percentile() 함수를 사용하여 데이터의 25%, 50%, 75%에 해당하는 사분위수를 구합니다. 이때 각각 1사분위수(quartile_1), 중앙값(q2), 3사분위수(quartile_3)로 할당됩니다.

# 사분위수를 기반으로 사분위범위(IQR, Interquartile Range)를 계산합니다. IQR은 3사분위수에서 1사분위수를 뺀 값으로, 데이터의 중간 50% 범위를 나타냅니다.

# 이상치의 하한(lower bound)과 상한(upper bound)을 계산합니다. 일반적으로 하한은 1사분위수에서 1.5배의 IQR을 뺀 값이고, 상한은 3사분위수에 1.5배의 IQR을 더한 값입니다.

# 수정된 부분에서는 상한(upper bound)을 quartile_3에 1.5배의 IQR을 더한 값으로 계산합니다.

# 이상치의 위치를 찾습니다. 이상치는 하한보다 작거나 상한보다 큰 데이터입니다. np.where() 함수를 사용하여 해당 조건을 만족하는 데이터의 위치를 찾습니다.

# 이상치의 위치를 반환합니다.
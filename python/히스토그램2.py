import numpy as np
import matplotlib.pyplot as plt

# 가상의 데이터 생성 (왼쪽으로 치우친 분포)
data = np.random.exponential(scale=2, size=90000)

# 로그 변환 적용
log_transformed_data = np.log1p(data)  # log1p 함수는 log(1 + x)를 계산

# 히스토그램 그리기
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(data, bins=50, color='blue', alpha=0.7)
plt.title('Original Data Histogram')

plt.subplot(1, 2, 2)
plt.hist(log_transformed_data, bins=50, color='green', alpha=0.7)
plt.title('Log Transformed Data Histogram')

plt.show()

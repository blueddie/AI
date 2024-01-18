import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

# 정규 분포에서 샘플링한 데이터
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=1000)

# 비정규 분포에서 샘플링한 데이터
non_normal_data = np.random.exponential(scale=1, size=1000)

# StandardScaler 적용
scaler = StandardScaler()
scaled_normal_data = scaler.fit_transform(normal_data.reshape(-1, 1))
scaled_non_normal_data = scaler.fit_transform(non_normal_data.reshape(-1, 1))

# 데이터와 정규 분포의 확률 밀도 함수 그리기
x = np.linspace(-3, 3, 1000)
pdf_normal = norm.pdf(x, loc=0, scale=1)

plt.figure(figsize=(12, 5))

# 정규 분포일 때
plt.subplot(1, 2, 1)
plt.title("Data from Normal Distribution")
plt.hist(normal_data, bins=30, density=True, color='blue', alpha=0.7)
plt.plot(x, pdf_normal, 'r-', label='Normal Distribution')
plt.xlabel("Values")
plt.ylabel("Density")
plt.legend()

# 비정규 분포일 때
plt.subplot(1, 2, 2)
plt.title("Data from Non-Normal Distribution")
plt.hist(non_normal_data, bins=30, density=True, color='blue', alpha=0.7)
plt.xlabel("Values")
plt.ylabel("Density")

plt.tight_layout()
plt.show()

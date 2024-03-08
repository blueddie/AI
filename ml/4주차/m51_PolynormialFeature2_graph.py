import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터
np.random.seed(777)
x = 2 * np.random.rand(100, 1) - 1                 # -1 부터 1까지 난수 생성
y = 3 * x**2 + 2 * x + 1 + np.random.randn(100, 1)  # np.random.randn(100, 1) -> 노이즈 추가? / y = 3x^2 +2x + 1 + 노이즈
# print(x)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly)

# 2. 모델
model = LinearRegression()
model2 = LinearRegression()

# model = RandomForestRegressor()
# model2 = RandomForestRegressor()


# 3. 훈련
model.fit(x, y)
model2.fit(x_poly, y)
print(x.shape, y.shape)

# 원본 데이터 시각화
plt.scatter(x, y, color='blue', label='원본 데이터')
plt.xlabel('x')
plt.xlabel('y')
plt.title('Polynormial Regression Example')

# 다항식 회귀 그래프 그리기
x_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
x_plot_poly = pf.transform(x_plot)
y_plot = model.predict(x_plot)
y_plot2 = model2.predict(x_plot_poly)
plt.plot(x_plot, y_plot, color='red', label='Polynormial Regression')
plt.plot(x_plot, y_plot2, color='green', label='그냥')

plt.legend()
plt.show()

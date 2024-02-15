from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape, x_test.shape)  #(60000, 28, 28) (10000, 28, 28)

# x = np.append(x_train, x_test, axis=0)
x = np.concatenate([x_train, x_test], axis=0)
print(x.shape)  #(70000, 28, 28)
x = x.reshape(70000, -1)
print(x.shape)  #(70000, 784)
#============================[실습]=======================================
# pca를 통해 0.95이상인 n_components는 몇 개?
# 0.95 이상
# 0.99 이상
# 0.999 이상
# 1.0일 때 몇 개?

# 힌트 np.argmax


# scaler = StandardScaler()
# x = scaler.fit_transform(x)

n_components = x.shape[1]   # 784

pca = PCA(n_components=n_components)
x_pca = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
# print(evr_cumsum)

over_95 = np.sum(evr_cumsum >= 0.95)
over_99 = np.sum(evr_cumsum >= 0.99)
over_999 = np.sum(evr_cumsum >= 0.999)
over_1 = np.sum(evr_cumsum >= 1.0)

print(f"0.95 이상 갯수: {over_95}")
print(f"0.99 이상 갯수: {over_99}")
print(f"0.999 이상 갯수: {over_999}")
print(f"1.0 이상 갯수: {over_1}")

# 0.95 이상 갯수: 631
# 0.99 이상 갯수: 454
# 0.999 이상 갯수: 299
# 1.0 이상 갯수: 72


print(np.argmax(evr_cumsum >= 0.95) + 1)    #154
print(np.argmax(evr_cumsum >= 0.99) + 1)    #331     
print(np.argmax(evr_cumsum >= 0.999) + 1)   #486
print(np.argmax(evr_cumsum >= 1.0) + 1)     #713

#==============================
# sum = 0
# for idx, evr in enumerate(evr_cumsum):
#     if evr >= 0.95:
#         # print("첫 번째로 0.9 이상인 값:", evr)
#         # print("해당 인덱스:", idx)
#         sum += 1
# print("0.95 이상 갯수", sum)

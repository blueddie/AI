import numpy as np

a = np.array(range(1, 11))
size = 5
print(len(a))

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1) :
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

print('a : ' , a)
print('a shape : ', a.shape)

bbb = split_x(a, size)
print('bbb: ' ,bbb)
print('bbb shape : ',bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]

print(x, y)
print(x.shape, y.shape)

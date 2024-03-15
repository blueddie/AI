import numpy as np

# 라벨과 해당하는 숫자 매핑
label_to_number = {
    'sport': 1,
    'business': 2,
    'politics': 3,
    'tech': 4,
    'entertainment': 5
}

# 주어진 라벨 배열
labels = ['business', 'entertainment', 'politics', 'sport', 'tech']

# 라벨을 숫자로 변환
numbers = [label_to_number[label] for label in labels]

# 숫자를 0부터 시작하도록 조정
adjusted_numbers = [number - 1 for number in numbers]

# 원-핫 인코딩 적용
one_hot_encoded = np.eye(5)[adjusted_numbers]

print(one_hot_encoded)
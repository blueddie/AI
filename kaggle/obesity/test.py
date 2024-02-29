import pandas as pd

# 예시 데이터프레임 생성
data = {
    'Height': [170, 165, 180, 155],
    'Weight': [65, 70, 75, 60],
    'Age': [30, 25, 35, 40]
}

df = pd.DataFrame(data)

# BMI 열 추가
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

print(df)
print("-------------------------------------")
#####################################################################
import pandas as pd

# 예시 데이터프레임 생성
data = {
    'Height': [1.7, 1.65, 1.8, 1.55],
    'Weight': [65, 70, 75, 60],
    'Age': [30, 25, 35, 40]
}

df = pd.DataFrame(data)

# BMI 열 추가 및 소수점 아래 세 번째 자리에서 버림
df = df.assign(BMI = (df['Weight'] / (df['Height'] ** 2)).round(2))

print(df)
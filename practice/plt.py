import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 예시 데이터프레임 생성
data = {'citric acid': [0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.7]}
df = pd.DataFrame(data)

# 스케일링을 적용하기 전 데이터 시각화
grouped_counts_before = df.groupby('citric acid').size()
plt.subplot(1, 2, 1)
plt.bar(grouped_counts_before.index, grouped_counts_before.values)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Before Scaling')

# 스케일링을 적용한 후 데이터 시각화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
grouped_counts_after = scaled_df.groupby('citric acid').size()
plt.subplot(1, 2, 2)
plt.bar(grouped_counts_after.index, grouped_counts_after.values)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('After Scaling')

plt.tight_layout()
plt.show()
'''
결측치 처리
1. 행 또는 열 삭제
2. 임의의 값
    평균 : mean
    중위 : median
    0 : fillna
    앞 값 : ffill
    뒷 값 : bfill
    특정 값 : 777 (뭔가 조건을 같이 넣는 게 좋겠지?)
3. 보간 :  interplate
4. 모델 : predict
5. 부스팅 계열 :  통상 결측치 이상치에 대해 자유롭다.

'''
import pandas as pd
from datetime import datetime
import numpy as np

dates = ["2/16/2024", "2/17/2024","2/18/2024",
         "2/19/2024", "2/20/2024", "2/21/2024"
         ]

dates = pd.to_datetime(dates)
print(dates)

print("==========================================================================")

ts = pd.Series([2, np.nan, np.nan
                , 8, 10, np.nan], index=dates)

print(ts)
print("==========================================================================")

ts = ts.interpolate()
print(ts) 
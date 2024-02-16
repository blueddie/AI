import pandas as pd

data = [
    ["삼성", "1000", "2000"]
    , ["현대", "1100", "3000"]
    , ["LG", "2000", "500"]
    , ["아모레", "3500", "6000"]
    , ["네이버", "100", "1500"]
 
]

index = ["031", "059", "033", "045", "023"] # 인덱스는 스트링, 연산할 때 쓰는 데이터가 아니다.
columns = ["종목명", "시가", "종가"]

df = pd.DataFrame(data=data, index=index, columns=columns)

print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500
print("================================")
# df[0]     # 에러
# df["031"] # 에러
print(df["종목명"]) # pandas 에서는 컬럼이 기준이다. 

# 아모레를 출력하고 싶다.
# print(df[4, 0])            # 에러
# print(df["종목명", "045"]) #  key 에러
print(df["종목명"]["045"])  # 아모레
#   판다스 열행 열행 열행~!~~~~~~~~!~!~!~!~!~!!~~!!~

# loc : 인덱스를 기준으로 행 데이터 추출
# iloc : 행 번호를 기준으로 행 데이터 추출
        # 인트 loc
print("============아모레 출력===============")
print(df.loc["045"])    # 인덱스 명
print(df.iloc[3])       # 인덱스 숫자
# print(df.loc[3])    # key 에러
print("=============네이버 출력=============")
print(df.loc["023"])
print(df.iloc[4])
print(df.iloc[-1])
print("=============아모레 시가(3500) 출력=============")
print(df.loc["045"].loc["시가"])
print(df.loc["045"].iloc[1])
print(df.iloc[3].iloc[1])
print(df.iloc[3].loc["시가"])

print(df.loc["045"][1]) # warning 하지만 가능
print(df.iloc[3][1])    # warning 하지만 가능

print(df.loc["045"]["시가"])
print(df.iloc[3]["시가"])

print(df.loc["045", "시가"])    # 3500
print(df.iloc[3, 1])            # 3500

print("================아모레와 네이버의 시가 출력===================")
print(df.iloc[3:5, 1])
print(df.iloc[[3,4], 1])
# print(df.iloc[3:5, "시가"]) # 에러
# print(df.iloc[[3,4], "시가"])   # 에러

# print(df.loc[3:5, "시가"])  # 에러
print(df.loc[["045", "023"], "시가"])


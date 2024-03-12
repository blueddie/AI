import json

# JSON 파일 경로
json_file_path = 'captions_train2017.json'

# JSON 파일 열기
with open(json_file_path, 'r') as f:
    data = json.load(f)

# JSON 데이터 구조 확인
print("Keys in the JSON file:", data.keys())

# captions 데이터에 접근
captions = data['annotations']
print("Number of captions:", len(captions))

# 첫 번째 이미지의 캡션 출력
first_caption = captions[0]
print("First caption:", first_caption['caption'])

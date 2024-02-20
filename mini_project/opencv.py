import cv2

# 이미지 파일 경로 지정
image_path = 'example.jpg'

# 이미지 파일을 흑백으로 읽어오기
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 이미지 윈도우에 표시
cv2.imshow('Gray Image', image)

# 아무 키나 누를 때까지 대기
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()
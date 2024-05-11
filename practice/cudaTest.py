# # import tensorflow as tf

# # print("cuDNN version:", tf.sysconfig.get_build_info()["cuda"]["cudnn_version"])

# import tensorflow as tf

# # GPU 사용 가능 여부 확인
# print("GPU Available:", tf.test.is_gpu_available())

# # 사용 가능한 GPU 목록 출력
# print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# import tensorflow as tf

# # TensorFlow GPU 사용 가능 여부 확인
# if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
#     from tensorflow.python.platform.build_info import build_info
#     print("1")
#     print(build_info.get("cuda_version"))
#     print("2")
#     print(build_info.get("cudnn_version"))
# else:
#     print("GPU is not available.")

########### gpu 연산 테스트 ################

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model

# GPU 사용 가능 여부 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 필요시 메모리 증가를 허용
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print("aaaaaaaaaaaaaa")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        print("aaaaaaaaaaaaaa")
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가를 설정해야만 합니다
        print(e)

# 간단한 컨볼루션 뉴럴 네트워크 정의
input_layer = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)

model = Model(inputs=input_layer, outputs=x)
model.summary()

# 임의의 데이터 생성 및 모델 실행
import numpy as np
input_data = np.random.random((1, 28, 28, 1)).astype('float32')
output = model(input_data)
print("Output shape:", output.shape)
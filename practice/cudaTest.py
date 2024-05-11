# # import tensorflow as tf

# # print("cuDNN version:", tf.sysconfig.get_build_info()["cuda"]["cudnn_version"])

# import tensorflow as tf

# # GPU 사용 가능 여부 확인
# print("GPU Available:", tf.test.is_gpu_available())

# # 사용 가능한 GPU 목록 출력
# print("GPU Devices:", tf.config.list_physical_devices('GPU'))

import tensorflow as tf

# TensorFlow GPU 사용 가능 여부 확인
if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    from tensorflow.python.platform.build_info import build_info
    print("1")
    print(build_info.get("cuda_version"))
    print("2")
    print(build_info.get("cudnn_version"))
else:
    print("GPU is not available.")
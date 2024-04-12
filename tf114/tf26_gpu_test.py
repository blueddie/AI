import tensorflow as tf

# tf.compat.v1.enable_eager_execution()
# 텐서플로 버전 :  1.14.0
# 즉시 실행 모드 : True

# 텐서플로 버전 :  2.9.0
# 즉시 실행 모드 : True
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

tf.compat.v1.disable_eager_execution()
# 텐서플로 버전 :  1.14.0
# 즉시 실행 모드 : False

# 텐서플로 버전 :  2.9.0
# 즉시 실행 모드 : False
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

print("텐서플로 버전 : ", tf.__version__)          
print("즉시 실행 모드 :", tf.executing_eagerly())   

# 텐서 1 코드를 텐서 2.9로 실행하겠다.
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        print(gpus[0])
    except RuntimeError as e:
        print(e)
else :
    print("gpu 없다!")
import tensorflow as tf
print(tf.__version__)   # 1.14.0
print(tf.executing_eagerly())   # False 즉시 실행 모드?

#  즉시 실행 모드 -> 텐서1의 그래프형태의 구성없이 자연스러운 파이썬 문법으로 실행시킨다.
#  즉시 실행 모드 켠다
tf.compat.v1.disable_eager_execution()    # 즉시 실행 모드 끈다. // 텐서플로 1.0 문법 // 디폴트 
# tf.compat.v1.enable_eager_execution()     # 즉시 실행 모드 켠다. // 텐스플로 2.0 사용 가능 

print(tf.executing_eagerly())   

hello = tf.constant('Hello World!!!')

sess = tf.compat.v1.Session()
print(sess.run(hello))

#   가상 환경       eager               사용 가능
    # 1.14        disable(디폴트)        가능     ★★★★
    # 1.14        enable                에러
    # 2.9.0       disable               가능     ★★★★
    # 2.9.0       enable(디폴트)         에러

# 텐서플로1에서는 eager 그냥 쓰면 됨 디폴트임 텐서플로2에서는 disable로 해줘야 함.

'''
Tensor 1은 '그래프 연산' 모드
Tenso 2는 '즉시 실행' 모드
tf.compat.v1.enable_eager_execustion()   #  즉시 실행 모드 켜
                    -> Tensor2의 디폴트

tf.compat.v1.disable_eager_execution()  #  즉시 실행 모드 꺼
                                -> 그래프 연산 모드로 돌아간다.
                                -> Tensor1 코드를 쓸 수 있다.
tf.executing_eagerly() # True면 즉시 실행 모드, -> Tensor2 코드만 써야 한다.
                        False 면 그래프 연산 모드 -> Tensor 1 코드를 쓸 수 있다.                                

'''
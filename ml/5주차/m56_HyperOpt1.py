import numpy as np
import hyperopt #pip install hyperopt
print(hyperopt.__version__)   # 0.2.7

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
search_space = {'x1' : hp.quniform('x1', -10, 10, 1),   # unifrom : 균등분포
                'x2' : hp.quniform('x2', -15, 15, 1)
                    #  hp.qniform(label, low, high, q)
                }

# hp.qunifom(label, low, high, q) : label로 지정된 입력 값 변수 검색 공간을
#                                      최소 값 low에서 최대값 high까지 q의 간격을 가지고 설정
# hp.qunifom(label, low, high) : 최소값 low에서 최대값 high까지 정규 분포 형태의 검색 공간 설정
# hp.randint(label, upper) : 0부터 최대값 upper까지  random 한 정수값으로 검색 공간 설정
# np.lohuniform(label, low, high) : exp(uniform(low, high))값을 반환하며 반환 값의 log변환 된 깂은 정규분포 형태를 가지는 검색 공간 설정

def objective_func(search_space):

    x1 = search_space['x1']
    x2 = search_space['x2']

    retrun_value =  x1**2 -20*x2

    return retrun_value

trial_val = Trials()

# print()

best = fmin(
    fn = objective_func,
    space=search_space,
    algo=tpe.suggest,   # 알고리즘 , 디폴트
    max_evals=20,   # 서치 횟수
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
    # rstate=333
)

print(best) #{'x1': 0.0, 'x2': 15.0}

print(trial_val.results)
# [{'loss': -216.0, 'status': 'ok'}, {'loss': -175.0, 'status': 'ok'}, {'loss': 129.0, 'status': 'ok'}, {'loss': 200.0, 'status': 'ok'}, {'loss': 240.0, 'status': 'ok'}, {'loss': -55.0, 'status': 'ok'}
#  , {'loss': 209.0, 'status': 'ok'}, {'loss': -176.0, 'status': 'ok'}, {'loss': -11.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, {'loss': 136.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}
#  , {'loss': 164.0, 'status': 'ok'}, {'loss': 321.0, 'status': 'ok'}, {'loss': 49.0, 'status': 'ok'},
#   {'loss': -300.0, 'status': 'ok'}, {'loss': 160.0, 'status': 'ok'}, {'loss': -124.0, 'status': 'ok'}, {'loss': -11.0, 'status': 'ok'}, {'loss': 0.0, 'status': 'ok'}]

print(trial_val.vals)
# 출처: https://ebbnflow.tistory.com/135 [Dev Log : 오늘의 삽질]

from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
 
# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])
 
print('x shape : ', x.shape) # (4,3)
print('y shape : ', y.shape) # (4,)
#  x  y
# 123 4
# 234 5
# 345 6
# 456 7
 
print(x)
print('-------x reshape-----------')
x = x.reshape((x.shape[0], x.shape[1], 1)) # (train data 개수, 한 주기의 길이, 몇 주기를 학습 시킬지)
print('x shape : ', x.shape)
print(x)
#  x        y
# [1][2][3] 4
# .....
 
# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3,1))) #(한 주기의 길이, 몇 주기를 학습 시킬지)
# DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
model.add(Dense(5))
model.add(Dense(1))
 
model.summary()
 
# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1)
 
x_input = array([6,7,8])
x_input = x_input.reshape((1,3,1))
 
yhat = model.predict(x_input)
print(yhat)





# EarlyStopping
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
 
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
 
print(x.shape) # (13,3)
print(y.shape) # (13,)
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x.shape) # (13,3,1)
 
model = Sequential()
model.add(LSTM(20, activation = 'relu', input_shape=(3,1)))
model.add(Dense(5))
model.add(Dense(1))
 
model.compile(optimizer='adam', loss='mse')
 
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
# loss값을 모니터해서 과적합이 생기면 100번 더 돌고 끊음
# mode=auto loss면 최저값이100번정도 반복되면 정지, acc면 최고값이 100번정도 반복되면 정지
# mode=min, mode=max
model.fit(x, y, epochs=1000, batch_size=1, verbose=2, callbacks=[early_stopping])
 
x_input = array([25,35,45]) # predict용
x_input = x_input.reshape((1,3,1))
 
yhat = model.predict(x_input)
print(yhat)



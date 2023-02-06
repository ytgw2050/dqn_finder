import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from keras import backend as K
import score
df1 = pd.read_excel('23_all_consider\경제 관련 자료 모음.xlsx')
df2 = pd.read_excel('23_all_consider\인구 관련 자료 모음.xlsx')
df1 = df1.iloc[:,1:]
df3 = df2.iloc[:,1:6]
df4 = df2.iloc[:,7:]
df = pd.concat([df1, df3,df4],axis=1)
#메인 변수 5개 - 소비자물가, 금리, 국내총생산, 신생아 출산인구, 합계 출산율
#Y값(레이블) - 수도권 인구 과밀화율
# X = df.iloc[:,0:5] #0~5까지 열, 즉 메인변수들
# y = df.iloc[:,5] #마지막 열, 즉 레이블
X = df.iloc[:395,:]
X.to_csv("test.csv")
y = df2.iloc[1:,6]
y.to_csv("y_.csv")
print(y) #
scaler = StandardScaler() #표준화. 평균이 0이고 분산이 1인 정규분포로 만드는 것


global loss_val_a
loss_val_a = []
loss_val_a = score.score('23_all_consider\경제 관련 자료 모음.xlsx')
loss_val_a += score.score('23_all_consider\인구 관련 자료 모음.xlsx')
loss_val_a = np.array([loss_val_a])
loss_val_a = np.reshape(loss_val_a, (1,22))
print(loss_val_a)
# loss_val_a = pd.DataFrame(loss_val_a)
X = X.to_numpy()
X = np.concatenate([X, loss_val_a])
X_norm = scaler.fit_transform(X)
# X_norm = scaler.fit_transform(X) #메인 변수들끼리 단위가 모두 다르기 때문에 표준화 시켜야 함.
numpY = np.empty((395,1)) #pandas라서 numpy형으로 바꿈
for i in range(1,395):
    print(y[i])
    numpY[i]=y[i]
loss_val_a = X_norm[-1,:]
X_norm = X_norm[:395,:]
# print(X_norm.shape)
#training set과 test set으로 나누기 #랜덤으로 스플릿
X_train,X_test,y_train,y_test=train_test_split(X_norm, numpY, test_size=0.1, random_state=42,shuffle=True)

# 모델 구조 정의하기
model = tf.keras.Sequential()  

# #입력 8개로부터 전달받는 12개 노드의 layer 생성
model.add(layers.Dense(128, input_shape=(22,),activation='sigmoid')) 
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(16,activation='relu')) #활성함수 relu
#회귀모형(regression) 구축을 위해서 linear 활성함수 사용
model.add(layers.Dense(1,activation='relu')) 
model.summary()
# 모델 구축하기
learning_rate = 1
momentum = 0.9
sgd = tf.keras.optimizers.Adam(learning_rate=100)
def r(y_test,y_pred):
    global loss_val_a
    return K.mean(K.square(y_pred-y_test)+K.square(y_pred-loss_val_a))
model.compile(
        loss=r,         # mean_squared_error(평균제곱오차)의 alias
        optimizer=sgd,   # 최적화 기법 중 하나
        metrics=['mae'],
)    # 실험 후 관찰하고 싶은 metric 들을 나열함. 
# hist = model.fit(
#     X_train, y_train,
#     batch_size=10,    
#     epochs=10000,       
#     validation_split=0.2,  
#     # callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mae', patience=0.01)], #과적합 방지용. loss가 100 epoch 동안 개선되지 않으면 학습 중단 
#     verbose=2) #학습 중 출력 문구 설정. 0이면 출력 X, 1이면 훈련 진행 막대, 2이면 미니배치마다 loss
hist = model.fit(
    X_train, y_train,
    batch_size=10000,    
    epochs=1000,       
    validation_split=0.2,  
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mae', patience=0.01)], #과적합 방지용. loss가 100 epoch 동안 개선되지 않으면 학습 중단 
    verbose=2)
# 모델 저장
model.save("test_dnn.h5") #test_dnn.h5라는 이름으로 모델 저장

scores = model.evaluate(X_test, y_test) #분리해둔 테스트 데이터로 모델 평가 후 scores 변수에 저장

# 관찰된 metric 값들을 확인함
for i in range(len(scores)):
    print("%s: %.2f" % (model.metrics_names[i], scores[i]))

#모델 손실 그래프 준비
fig, loss_ax = plt.subplots(figsize=(15, 5))

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train_mse')   # 훈련데이터의 loss (즉, mse)
loss_ax.plot(hist.history['val_loss'], 'r', label='test_mse') # 검증데이터의 loss (즉, mse)

acc_ax.plot(hist.history['mae'], 'b', label='train_mae')   # 훈련데이터의 mae
acc_ax.plot(hist.history['val_mae'], 'g', label='val_mae') # 검증데이터의 mae

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('mean_absolute_error')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# # 모델 불러오기
# loaded_model = load_model("test_dnn.h5")
# score = model.evaluate(X_test, y_test)
# print('test_loss: ', score[0])
# print('test_mae: ', score[1])
# hidden_2 = model.layers[0]
# weights, biases = hidden_2.get_weights()
# print(weights)
# print(biases)
X_test = list(X_test)[0]
mg = model.predict(np.array([loss_val_a]))
print(X_test)
m = []
n = []
k = 0 
def c(y_test,y_pred):
    global loss_val_a
    return abs(y_pred-y_test)*5+abs(y_pred-mg[0][0])
while k < 100:
    X_test[6] = k
    m.append(k)
    n.append(c(model.predict(np.array([X_test]))[0][0],y_test[0]))
    k += 0.1
plt.plot(m,n)
plt.show()

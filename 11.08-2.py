from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('C:\data\sonar3.csv', header=None)

# 음파 관련 속성을 x로
x = df.iloc[:, 0:60]
# 광물의 종류를 y로
y = df.iloc[:, 60]

k = 7
# kfold 함수를 불러옴, 샘플이 치우치지않게 섞어줌
kfold = KFold(n_splits=k, shuffle=True)
# 정확도가 채워질 빈 리스트 생성
acc_score = []


def model_fn():
    model = Sequential()
    model.add(Dense(16, input_dim=60, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


for train_index, test_index in kfold.split(x):
    x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = model_fn()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=200, batch_size=10, verbose=0)

    accuracy = model.evaluate(x_test, y_test)[1]
    acc_score.append(accuracy)

avg_acc_score = sum(acc_score) / k

print('정확도:', acc_score)
print('정확도 평균:', avg_acc_score)

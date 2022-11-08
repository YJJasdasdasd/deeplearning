from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv('C:\data\sonar3.csv', header=None)

df.head()

# 광석과 암석의 샘플 갯수

df[60].value_counts()

x = df.iloc[:, 0:60]
y = df.iloc[:, 60]


# 모델 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x, y, epochs=200, batch_size=10)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=200, batch_size=10)
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])

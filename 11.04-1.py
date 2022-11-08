from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\data\pima-indians-diabetes3.csv")

df.head()

df["diabetes"].value_counts()

# 일반인 500명
# 당뇨병 환자 268명

df.describe()  # 정보별 특징 count(샘플 수) mean(평균) std(표준편차) min(최솟값) 백분위수 max(최댓값)

df.corr()

# 그래프의 색상구성을 정합니다
colormap = plt.cm.gist_heat
# 그래프의 크기를 정합니다.
plt.figure(figsize=(12, 12))

sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5,
            cmap=colormap, linecolor='white', annot=True)
plt.show()

plt.hist(x=[df.plasma[df.diabetes == 0], df.plasma[df.diabetes == 1]],
         bins=30, histtype='barstacked', label=['normal', 'diabetes'])
plt.legend()

plt.hist(x=[df.bmi[df.diabetes == 0], df.bmi[df.diabetes == 1]],
         bins=30, histtype='barstacked', label=['normal', 'diabetes'])
plt.legend()

x = df.iloc[:, 0:8]
y = df.iloc[:, 8]


model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', name='Dense_1'))
model.add(Dense(8, activation="relu", name='Dense_2'))
model.add(Dense(1, activation='sigmoid', name='Dense_3'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x, y, epochs=1000, batch_size=5)

import tensorflow
import pandas
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#데이터 호출, MinMaxScaler를 통한 정규화, train, test 분리
df = pandas.read_csv("./covtype.csv")
x = df[df.columns[:55]]
y = df.Cover_Type

norm = x[x.columns[:10]]    #정규화 할 column들 선택
minmax_scaler = preprocessing.MinMaxScaler().fit(norm)
x_norm = minmax_scaler.transform(norm)
norm_cols = pandas.DataFrame(x_norm, index=norm.index, columns=norm.columns)
x.update(norm_cols)
print(x.head())
print(x["Elevation"].describe())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=10)

#모델 정의
model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
    tensorflow.keras.layers.Dense(64, activation="relu"),
    tensorflow.keras.layers.Dense(8, activation="softmax")
])

#모델 컴파일 및 훈련
model.compile(optimizer=tensorflow.keras.optimizers.Adam(0.01),
              loss="sparse_categorical_"
                   "crossentropy",
              metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test))
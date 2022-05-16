"""
-환경-
os : windows 10
gpu : RTX 3060
tensorflow : 2.4.0
python : 3.8
cuda : 11.0
cudnn : 8.0.5
anaconda : 4.11.0
numpy : 1.19.5
pandas : 1.3.1
sklearn : 0.24.2
matplotlib : 3.4.2
cv2 : 4.5.5

-데이터-
IMDB데이터셋은 영화 리뷰에 대한 데이터 5만개로 이루어져 있다.

25,000개는 훈련용, 나머지 25,000개는 테스트용으로 나누어져 있으며, 각각 50%씩 긍정, 부정 리뷰로 이루어져 있다.

이 데이터는 tensorflow.keras.datasets에서 구할 수 있으며 이미 전처리 되어 각 리뷰가 숫자로 변환되어 있다.

-목적-
Bidirectional LSTM을 구성하여 영화 리뷰가 긍정인지 부정인지 판별하는 모델 학습

-학습 결과-
훈련 데이터셋에 약 97%, 테스트 데이터셋에 약 86%의 성능을 보였다.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow
import numpy

#값 초기화
unique_words_num = 10000
maxlen = 200
embedding_size = 128
batch_size = 128

#데이터셋 호출
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.imdb.load_data(num_words=unique_words_num)

x_train = tensorflow.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tensorflow.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = numpy.array(y_train)
y_test = numpy.array(y_test)

#모델 정의
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Embedding(input_dim=unique_words_num, output_dim=embedding_size, input_length=maxlen))
model.add(tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(64)))
model.add(tensorflow.keras.layers.Dense(1, activation="sigmoid"))

#모델 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test), verbose=1)

#모델 평가
print("훈련 데이터셋")
(loss, accuracy) = model.evaluate(x_train, y_train, verbose=0)
print("loss = {:.4f} and accuracy = {:.4f}%".format(loss, accuracy*100))
print("테스트 데이터셋")
(loss, accuracy) = model.evaluate(x_test, y_test, verbose=0)
print("loss = {:.4f} and accuracy = {:.4f}%".format(loss, accuracy*100))
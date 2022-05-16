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
GRU를 2-layer로 구성하여 영화 리뷰가 긍정인지 부정인지 판별하는 모델 학습

-학습 결과-
훈련 데이터셋에 약 98%, 테스트 데이터셋에 약 82%의 성능을 보였다.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow
import numpy

#값 초기화
tensorflow.random.set_seed(10)
numpy.random.seed(10)

batch_size = 128
total_words = 10000
max_review_len = 80
embedding_len = 100

#데이터셋 호출
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.imdb.load_data(num_words=total_words)

x_train = tensorflow.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = tensorflow.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

train_data = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(25000).batch(batch_size=batch_size, drop_remainder=True)
test_data = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.shuffle(25000).batch(batch_size=batch_size, drop_remainder=True)
print("x_train shape : ", x_train.shape, tensorflow.reduce_max(y_train), tensorflow.reduce_min(y_train))
print("x_test shape : ", x_test.shape)

#GRU를 이용한 모델 정의
class GRU(tensorflow.keras.Model):
    def __init__(self, units):
        super(GRU, self).__init__()

        self.embedding = tensorflow.keras.layers.Embedding(input_dim=total_words, output_dim=embedding_len, input_length=max_review_len)
        self.gru = tensorflow.keras.Sequential([
            tensorflow.keras.layers.GRU(units=units, dropout=0.2, return_sequences=True, unroll=True),
            tensorflow.keras.layers.GRU(units=units, dropout=0.2)
        ])
        self.outlayer = tensorflow.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        x = self.gru(x)
        x = self.outlayer(x)
        prob = tensorflow.sigmoid(x)

        return prob

#모델 훈련
units = 64
epochs = 5

model = GRU(units)
model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
              loss=tensorflow.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.fit(train_data, epochs=epochs, validation_data=test_data, verbose=1)

#모델 평가
print("훈련 데이터셋")
(loss, accuracy) = model.evaluate(train_data, verbose=0)
print("loss = {:.4f} and accuracy = {:.4f}%".format(loss, accuracy*100))
print("테스트 데이터셋")
(loss, accuracy) = model.evaluate(test_data, verbose=0)
print("loss = {:.4f} and accuracy = {:.4f}%".format(loss, accuracy*100))
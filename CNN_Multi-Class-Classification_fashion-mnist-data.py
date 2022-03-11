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

-데이터-
레코드 수 : 60000 + 10000(각각 train과 test데이터)
데이터 형태 : 입력 데이터는 이미지를 나타내는 28 * 28 * 1 크기의 0~255값을 가지는 넘파이 배열, 정답 데이터는 0~9까지의 정수 값을 가지는 배열
데이터 출처 : tensorflow 내부의 케라스에서 제공

-목적-
CNN을 통해 입력으로 들어온 이미지가 어떠한 클래스에 속하는지 분류

-학습 결과-
테스트 셋에 대해 약 92%의 정확도를 보임
"""
import tensorflow
import matplotlib.pyplot

#데이터 호출 및 train, test데이터 구별
fashion_mnist = tensorflow.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#matplotlib를 이용하여 데이터가 어떤 모양인지 살펴보기
for i in range(25):
    matplotlib.pyplot.subplot(5, 5, i + 1)
    matplotlib.pyplot.grid(False)
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.yticks([])
    matplotlib.pyplot.imshow(X_train[i], cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.show()

#데이터 전처리
X_train = X_train.reshape((-1, 28, 28, 1)) / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)) / 255.0

#모델 생성
model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tensorflow.keras.layers.MaxPool2D((2, 2), strides=2),
    tensorflow.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tensorflow.keras.layers.MaxPool2D((2, 2), strides=2),
    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(128, activation='relu'),
    tensorflow.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#모델 훈련 및 성능 평가
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
model.evaluate(X_test, y_test, verbose=1)

#모델 정보 json파일로 저장
model_json = model.to_json()
with open('./model.json', 'w') as json_file:
    json_file.write(model_json)

#학습된 모델의 weights저장
model.save_weights('model.h5')
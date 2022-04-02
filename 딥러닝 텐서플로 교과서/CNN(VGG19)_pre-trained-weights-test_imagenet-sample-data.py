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
kaggle에서 VGG19의 pre-train된 weights를 가져왔다

또한 분류가 올바르게 되는지 테스트 하기 위한 이미지넷 데이터 3장을 사용하였다

-목적-
VGG19의 구조를 파악하고 학습된 모델이 예측을 잘하는지 확인

-결과-
3개의 샘플데이터에 대해 정확히 분류함을 확인함
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
import cv2
import matplotlib.pyplot
import numpy

#VGG19 모델 정의
class VGG19(tensorflow.keras.models.Sequential):
    def __init__(self, input_shape):
        super().__init__()

        self.add(tensorflow.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
        self.add(tensorflow.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(tensorflow.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(tensorflow.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(tensorflow.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(tensorflow.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(tensorflow.keras.layers.Flatten())
        self.add(tensorflow.keras.layers.Dense(4096, activation='relu'))
        self.add(tensorflow.keras.layers.Dropout(0.5))
        self.add(tensorflow.keras.layers.Dense(4096, activation='relu'))
        self.add(tensorflow.keras.layers.Dropout(0.5))
        self.add(tensorflow.keras.layers.Dense(1000, activation='softmax'))

        self.compile(optimizer=tensorflow.keras.optimizers.Adam(0.003),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

#VGG19 모델 호출 및 정보 출력
model = VGG19(input_shape=(224, 224, 3))
model.summary()

#사전 훈련된 모델의 weights 적용
model.load_weights("../Datasets/pre-trained-weights/vgg19_weights_tf_dim_ordering_tf_kernels.h5")

#샘플 사진 3개에 대해 예측 테스트
classes = {282:"cat",
           681:"notebook, notebook computer",
           970:"alp"}

#image = cv2.imread("../Datasets/imagenet-samples/cat.jpg")
#image = cv2.imread("../Datasets/imagenet-samples/labtop.jpg")
image = cv2.imread("../Datasets/imagenet-samples/starrynight.jpeg")
image = cv2.resize(image, (224, 224))
matplotlib.pyplot.figure()
matplotlib.pyplot.imshow(image)
image = image[numpy.newaxis, :]
predicted_value = model.predict(image)
matplotlib.pyplot.title(classes[numpy.argmax(predicted_value, axis=-1)[0]])
matplotlib.pyplot.show()
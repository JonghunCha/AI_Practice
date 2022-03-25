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
레코드 수 : 25000(train : 16000, validation : 4000, test : 5000, 반은 강아지, 반은 고양이 사진)
데이터 출처 : 캐글에서 제공, 라벨이 있는 데이터들만 사용

-목적-
AlexNet CNN을 통해 입력으로 들어온 이미지가 강아지 사진인지 고양이 사진인지를 구별

-학습 결과-
학습데이터의 수에 비해 모델이 너무 커서 제대로 수렴하는 경우를 찾기가 매우 힘들었음

레이어 몇 개를 줄이니 학습이 진행되는 것을 알 수 있었음

따라서 데이터의 수에 따라 모델의 크기를 적절히 조절해야 함을 알 수 있었음
"""
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow

#AlexNet 모델 정의
class AlexNet(tensorflow.keras.models.Sequential):
    def __init__(self, input_shape):
        super().__init__()

        self.add(tensorflow.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu', input_shape=input_shape))
        self.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))
        self.add(tensorflow.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))
        self.add(tensorflow.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last'))
        self.add(tensorflow.keras.layers.Flatten())
        self.add(tensorflow.keras.layers.Dense(4096, activation='relu'))
        self.add(tensorflow.keras.layers.Dense(4096, activation='relu'))
        self.add(tensorflow.keras.layers.Dense(1000, activation='relu'))
        self.add(tensorflow.keras.layers.Dense(2, activation='softmax'))

        self.compile(optimizer=tensorflow.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#모델 생성
model = AlexNet((227, 227, 3))
model.summary()

#데이터 호출 및 전처리
train_dir = "../Datasets/dogs-vs-cats/train/"
valid_dir = "../Datasets/dogs-vs-cats/validation/"
test_dir = "../Datasets/dogs-vs-cats/test/"

train = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_generator = train.flow_from_directory(train_dir,
                                            target_size=(227, 227),
                                            color_mode="rgb",
                                            batch_size=32,
                                            seed=1,
                                            shuffle=True,
                                            class_mode='categorical')

valid = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

valid_generator = valid.flow_from_directory(valid_dir,
                                            target_size=(227, 227),
                                            color_mode="rgb",
                                            batch_size=32,
                                            seed=1,
                                            shuffle=True,
                                            class_mode='categorical')

test = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

test_generator = test.flow_from_directory(valid_dir,
                                            target_size=(227, 227),
                                            color_mode="rgb",
                                            batch_size=32,
                                            seed=1,
                                            shuffle=True,
                                            class_mode='categorical')

train_num = train_generator.samples
valid_num = valid_generator.samples
test_num = test_generator.samples

#텐서보드 콜백 설정
log_dir = "./log"
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)

#모델 학습 및 평가
model.fit(train_generator, epochs=100, steps_per_epoch=train_num//32, validation_data=valid_generator, validation_steps=valid_num//32, callbacks=[tensorboard_callback], verbose=1)
model.evaluate(test_generator, verbose=1)

#모델 정보 json파일로 저장
model_json = model.to_json()
with open('./model.json', 'w') as json_file:
    json_file.write(model_json)

#학습된 모델의 weights저장
model.save_weights('model.h5')
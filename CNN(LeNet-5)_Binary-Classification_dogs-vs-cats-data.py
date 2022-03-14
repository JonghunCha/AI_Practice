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
LeNet-5 기반의 CNN을 통해 입력으로 들어온 이미지가 강아지 사진인지 고양이 사진인지를 구별

-학습 결과-
테스트 셋에 대해 약 82%의 정확도를 보임
"""
import tensorflow

#LeNet-5 클래스 생성
class LeNet(tensorflow.keras.models.Sequential):
    def __init__(self, input_shape, class_nums):
        super().__init__()

        self.add(tensorflow.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding='same'))
        self.add(tensorflow.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(tensorflow.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
        self.add(tensorflow.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(tensorflow.keras.layers.Flatten())
        self.add(tensorflow.keras.layers.Dense(120, activation='relu'))
        self.add(tensorflow.keras.layers.Dense(84, activation='relu'))
        self.add(tensorflow.keras.layers.Dense(2, activation='softmax'))

        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#LeNet-5 모델 생성
model = LeNet((100, 100, 3), 2)
model.summary()

#데이터 호출
train = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                rotation_range=10,
                                                                width_shift_range=0.1,
                                                                height_shift_range=0.1,
                                                                shear_range=0.1,
                                                                zoom_range=0.1)
train_generator = train.flow_from_directory('./train',
                                            target_size=(100, 100),
                                            color_mode='rgb',
                                            batch_size=32,
                                            seed=1,
                                            shuffle=True,
                                            class_mode='categorical')

valid = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                rotation_range=10,
                                                                width_shift_range=0.1,
                                                                height_shift_range=0.1,
                                                                shear_range=0.1,
                                                                zoom_range=0.1)
valid_generator = train.flow_from_directory('./validation',
                                            target_size=(100, 100),
                                            color_mode='rgb',
                                            batch_size=32,
                                            seed=1,
                                            shuffle=True,
                                            class_mode='categorical')

test = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                rotation_range=10,
                                                                width_shift_range=0.1,
                                                                height_shift_range=0.1,
                                                                shear_range=0.1,
                                                                zoom_range=0.1)
test_generator = train.flow_from_directory('./test',
                                            target_size=(100, 100),
                                            color_mode='rgb',
                                            batch_size=32,
                                            seed=1,
                                            shuffle=True,
                                            class_mode='categorical')

train_num = train_generator.samples
valid_num = valid_generator.samples
test_num = test_generator.samples

#모델 학습 및 평가
model.fit(train_generator, epochs=100, steps_per_epoch=train_num//32, validation_data=valid_generator, validation_steps=valid_num//32, verbose=1)
model.evaluate(test_generator, verbose=1)

#모델 정보 json파일로 저장
model_json = model.to_json()
with open('./model.json', 'w') as json_file:
    json_file.write(model_json)

#학습된 모델의 weights저장
model.save_weights('model.h5')
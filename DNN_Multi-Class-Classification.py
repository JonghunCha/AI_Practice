"""
-환경-
os : windows 10
tensorflow : 2.4.0
python : 3.8
anaconda : 4.10.1
pandas : 1.3.1
sklearn : 0.24.2
matplotlib : 3.4.2

-데이터-
자동차 가격, 자동차 유지 비용, 자동차 문 개수, 수용 인원, 수하물 용량, 안정성, 자동차의 상태 정보로 이루어져 있음

-목적-
자동차 가격, 자동차 유지 비용, 자동차 문 개수, 수용 인원, 수하물 용량, 안정성 정보를 이용하여 자동차의 상태 정보를 예측

-데이터 전처리-
각 column에 올 수 있는 모든 정보에 대해 one-hot encoding을 적용
그 결과 input으로 사용되는 데이터는 21차원, output으로 사용되는 데이터는 4차원의 데이터로 구성됨

-학습 결과-
테스트 셋에 대해 score는 0.038, acuuracy는 0.988의 성능을 보임

-비고-
데이터는 저작권이 확인되지 않아 게시하지 못하였음
"""
import tensorflow
import pandas
import sklearn.model_selection
import matplotlib.pyplot

#데이터 호출
cols = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety', 'output']
cars = pandas.read_csv('car_evaluation.csv', names=cols, header=None)

#데이터셋 분포 확인(확인 결과 output의 분포는 70:22:4:4정도로 이루어져 있음)
cars.output.value_counts().plot(kind='pie', autopct='%0.05f%%', colors=['lightblue', 'lightgreen', 'orange', 'pink'])
matplotlib.pyplot.savefig("dataset.png")
matplotlib.pyplot.show()

#one-hot인코딩 적용
price = pandas.get_dummies(cars.price, prefix='price')
maint = pandas.get_dummies(cars.maint, prefix='maint')
doors = pandas.get_dummies(cars.doors, prefix='doors')
persons = pandas.get_dummies(cars.persons, prefix='persons')
lug_capacity = pandas.get_dummies(cars.lug_capacity, prefix='lug_capacity')
safety = pandas.get_dummies(cars.safety, prefix='safety')
labels = pandas.get_dummies(cars.output, prefix='condition')

X = pandas.concat([price, maint, doors, persons, lug_capacity, safety], axis=1)
y = labels.values

#train, test데이터 분리
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

#모델 생성 및 컴파일
input_layer = tensorflow.keras.layers.Input(shape=(X.shape[1],))
dense_layer1 = tensorflow.keras.layers.Dense(15, activation='relu')(input_layer)
dense_layer2 = tensorflow.keras.layers.Dense(10, activation='relu')(dense_layer1)
output = tensorflow.keras.layers.Dense(y.shape[1], activation='softmax')(dense_layer2)

model = tensorflow.keras.models.Model(inputs=input_layer, outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()

#모델 훈련
history = model.fit(X_train, y_train, batch_size=8, epochs=50, verbose=1, validation_split=0.2)

#test데이터를 통한 모델 평가
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score : ", score[0])
print("Test Accuracy : ", score[1])

#모델 정보 json파일로 저장
model_json = model.to_json()
with open('./model.json', 'w') as json_file:
    json_file.write(model_json)

#학습된 모델의 weights저장
model.save_weights('model.h5')
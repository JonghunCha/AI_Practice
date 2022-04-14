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
레코드 수 : train(10886), test(6493)이다.
데이터 출처 : Kaggle

해당 데이터의 본래 목적은 train데이터로 학습을 한 뒤, 학습한 모델로 test데이터에 대한 자전거 대여량을 예측하고 이를 sampleSubmission의 형식과 같게 저장하는 것이다.

그러나 본 코드는 kaggle에 제출할 용도가 아니기에 train의 80%로만 훈련을 하고, 나머지 20%를 검증용으로 사용하였다.

-목적
train데이터와 GradientBoosting을 이용하여 각 시간 별 자전거 대여량을 예측한다.

-학습 결과-
몇 번의 파라미터 조정으로 실험한 결과 train에서는 약 97%, validation에서는 약 88%의 정확도를 보였다.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

#1.데이터 호출
train = pd.read_csv("E:/AI_Practice/Datasets/bike-sharing-demand/train.csv")

#2.데이터 확인
print(train.info())
print("\n")

#3.데이터 전처리
#3-1.object로 저장 된 "datatime"column을 datetime64로 바꾸어 준다.
train["datetime"] = pd.to_datetime(train["datetime"])
print(train.info())
print("\n")

#3-2."datetime"에서 사용할 시간 정보를 추출하여 새로운 column생성
#이 때 년도, 월, 날짜, 시, 분, 초 중 자전거 대여량 예측에는 월, 시가 유효한 정보를 가지고 있다고 판단하여 둘 만 추출하였다.
#또한 이 외에 요일 정보도 주요 정보라 생각하여 아래와 같이 월, 시, 요일 column을 추가하였다.
train["month"] = train["datetime"].dt.month
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek
print(train.info())
print("\n")

#3-3.범주형 변수 one-hot encoding
train = pd.get_dummies(data=train, columns=["season", "holiday", "workingday", "weather"], prefix=["season", "holiday", "workingday", "weather"])
print(train.info())
print("\n")

#3-4.training에 사용할 변수 선택
#컬럼의 내용에서 자전거 대여량에 영향을 미칠 것이라 예상하는 후보들을 아래와 같이 골랐다.
#temp, humidity, windspeed, month, hour, dayofweek, season(1~4), holiday(0~1), weather(1~4)
train.drop(["datetime", "atemp", "casual", "registered", "workingday_0", "workingday_1"], axis=1, inplace=True)
print(train.info())
print("\n")

#4.Gradient Boosting 모델 학습 및 평가
#4-1.데이터 분할
#학습에 사용되는 데이터는 DataFrame이 아닌 array형태여야 하기에 values를 추출
x_train = train.drop("count", axis=1).values
y_train = train["count"].values

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=10)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

#4-2.모델 생성 및 학습
model = GradientBoostingRegressor(n_estimators=2000,
                                  max_depth=5,
                                  min_samples_split=20,
                                  min_samples_leaf=20,
                                  random_state=10,
                                  verbose=1)
model.fit(x_train, y_train)

#4-3.모델 성능 평가
train_score = model.score(x_train, y_train)
val_score = model.score(x_val, y_val)
print("train score : %f" %train_score)
print("validation score : %f" %val_score)
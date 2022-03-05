"""
-환경-
os : windows 10
python : 3.8
anaconda : 4.11.0
numpy : 1.19.5
pandas : 1.3.1
sklearn : 0.24.2
matplotlib : 3.4.2

-데이터-
kaggle의 weather dataset을 이용
해당 데이터 셋은 전 세계 여러 기상 관측소에서 측정한 여러가지 기상 정보들로 이루어져 있음(강수량, 강설량, 풍속, 기온 등등)

-목적-
최저기온을 보고 최고기온을 예측

-학습 결과-
MSE(Mean Squared Error)의 경우 약 18
RMSE(Rooted Mean Squared Error)의 경우 약 4.24
위와 같은 오차를 보였다
"""
import pandas
import numpy
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#데이터 호출
weather = pandas.read_csv('./weather.csv')

#MinTemp와 MaxTemp간의 관계 시각화
weather.plot(x='MinTemp', y='MaxTemp', style='o')
matplotlib.pyplot.title('MinTemp & MaxTemp')
matplotlib.pyplot.xlabel('MinTemp')
matplotlib.pyplot.ylabel('MaxTemp')
matplotlib.pyplot.show()

#사용할 칼럼만 추출해 각각 X와 y로 지정 및 train, test데이터 분할
X = weather['MinTemp'].values.reshape(-1,1)
y = weather['MaxTemp'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#모델 생성 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

#테스트 데이터와 학습된 모델의 회귀선 시각화
y_predict = model.predict((X_test))
matplotlib.pyplot.scatter(X_test, y_test, color='grey')
matplotlib.pyplot.plot(X_test, y_predict, color='red', linewidth=1)
matplotlib.pyplot.show()

#모델 평가
print("MSE(Mean Squared Error) : ", metrics.mean_squared_error(y_test, y_predict))
print("RMSE(Rooted Mean Squared Error) : ", numpy.sqrt(metrics.mean_squared_error(y_test, y_predict)))
"""
-환경-
os : windows 10
python : 3.8
anaconda : 4.11.0
pandas : 1.3.1
sklearn : 0.24.2

-데이터-
레코드 수 : 150
칼럼 수 : 5개(4개는 속성, 1개는 분류 class)
분류 클래스의 수 : 3개(각각 50개의 레코드를 가짐)
데이터 링크 : https://archive.ics.uci.edu/ml/datasets/iris

-목적-
꽃받침(sepal)과 꽃잎(petal)의 길이 폭 정보를 입력으로 하여 꽃의 종류를 구별한다(분류 작업)

-데이터 전처리-
sklearn.preprocessing의 StandardScaler를 이용하여 입력으로 주어지는 길이 정보를 평균이 0, 표준편차가 1이 되도록 변환하였다.
(이는 특정 입력에 민감하게 반응하지 않도록 하기 위함이다)

-학습 결과-
K를 10으로 설정하였을 때 0.967의 정확도를 보임

-비고-
KNN은 K의 크기에 따라 성능이 달라질 수 있다.
따라서 적절한 K를 잘 찾는 것이 KNN사용의 핵심이다.
"""
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#데이터 호출
cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal=width', 'Class']
iris = pandas.read_csv('iris.data', names=cols)

#train, test데이터 분리
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)

#모델 생성 및 훈련
KNN = KNeighborsClassifier(n_neighbors=10)  #K=50인 KNN생성
KNN.fit(X_train, y_train)

#모델 정확도 확인
y_predict = KNN.predict(X_test)
print("Test Accuracy : ", accuracy_score(y_test, y_predict))
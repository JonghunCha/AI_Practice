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

-학습 결과-
테스트 데이터셋에 대하여 1.0의 정확도로 100퍼센트 정확하게 분류함을 알 수 있었다

-비고-
iris data는 선형으로 분류 가능한 데이터라 linear커널을 사용하였다.
만약 주어진 데이터가 선형으로 분류가 힘든 경우 가우시안 RBF커널 혹은 다항식 커널을 사용하여 데이터를 고차원으로 매핑한 뒤 분류를 하면 된다.
"""
import pandas
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#데이터 호출
cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal=width', 'Class']
iris = pandas.read_csv('iris.data', names=cols)

#train, test데이터 분리
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#모델 생성 및 훈련
SVM = svm.SVC(kernel='linear', C=1.0, gamma=0.5)
SVM.fit(X_train, y_train)

#모델 정확도 확인
y_predict = SVM.predict(X_test)
print("Test Accuracy : ", accuracy_score(y_test, y_predict))
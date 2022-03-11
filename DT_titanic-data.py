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
레코드 수 : 891
칼럼 수 : 12개
분류 클래스의 수 : 2개(생존, 사망 여부를 나타내는 Survived칼럼을 예측)
데이터 링크 : https://www.kaggle.com/c/titanic

-데이터 전처리-
총 12개의 칼럼에서 사용 할 칼럼 7개를 추출하였다
또한, 기존 데이터에 성별은 male, female로 적혀있는데, 이를 0과 1로 바꾸었다

-목적-
생존여부를 파악하는데 유의미한 칼럼들을 이용하여 해당 승객이 생존자인지 사망자인지 예측

-학습 결과-
train, test데이터를 어떻게 나누냐에 따라 결과가 달라졌으나 대부분 70%~85%정도의 정확도를 보였다

-비고-
위 링크에서 제공하는 데이터는 gender_submission, test, train 3개이나
test파일에는 정답 라벨링이 되어있지 않아 train파일을 train과 test로 나누어서 사용함
"""
import pandas
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

#데이터 호출
titanic_train = pandas.read_csv('./train.csv', index_col='PassengerId')

#데이터 전처리
titanic_train = titanic_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]   #예측작업에 필요한 컬럼만 추출
titanic_train['Sex'] = titanic_train['Sex'].map({'male': 0, 'female': 1})   #성별 데이터를 남자는 0, 여자는 1로 매핑
titanic_train = titanic_train.dropna()  #값이 없는 데이터는 삭제
X = titanic_train.drop('Survived', axis=1)
y = titanic_train['Survived']

#train, test데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Decision Tree모델 생성
model = tree.DecisionTreeClassifier()

#모델 훈련
model.fit(X_train, y_train)

#모델 정확도 확인
y_predict = model.predict(X_test)
print("Test Accuracy : ", accuracy_score(y_test, y_predict))
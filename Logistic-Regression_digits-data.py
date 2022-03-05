"""
-환경-
os : windows 10
python : 3.8
anaconda : 4.11.0
sklearn : 0.24.2

-데이터-
sklearn에서 제공하는 digits데이터 사용(사람이 숫자를 손글씨로 적은 데이터)
digits데이터는 8*8이미지 1797개로 구성되어 있음

-목적-
이미지를 보고 해당 손글씨가 어떤 숫자를 적은 것인지 예측
0~9 총 10개의 클래스로 분류

-학습 결과-
약 97%의 정확도를 보임
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#데이터 호출 및 데이터 형태 확인
digits = load_digits()
print("Image Data Shape : ", digits.data.shape)
print("Label Data Shape : ", digits.target.shape)

#train, test데이터 분리
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

#Logistic Regreesion모델 생성 및 훈련
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

#모델 성능 확인
score = model.score(X_test, y_test)
print(score)
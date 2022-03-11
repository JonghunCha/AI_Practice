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
레코드 수 : 440
칼럼 수 : 8개(Channel Region Fresh Milk Grocery Frozen Detergents_Paper Delicassen)
앞에서 부터 각각 고객 채널(호텔/레스토랑/카페 등), 고객 지역, 신선한 제품 지출, 유제품 지출, 식료품 지출, 냉동 제품 지출, 세제 및 종이 지출, 조제 식품 지출을 의미

-데이터 전처리-
Channerl과 Region은 명목형 데이터라 one-hot 인코딩 방식으로 변형 후 사용

-목적-
비슷한 성격의 데이터끼리 서로 묶는 클러스터링 작업

-학습 결과-
K = 5 또는 6정도에서 Sum of Squared Distance(SSD) 값이 어느정도 수렴한 것으로 볼 수 있었다

-비고-
데이터는 책에서 제공하는 데이터였는데, 원래 출처를 알아내지 못함
"""
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot

#데이터 호출
sales = pandas.read_csv('./sales data.csv')

#명목형 데이터와 연속형 데이터 구분
categorical_feature = ['Channel', 'Region']
continuous_feature = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

#명목형 데이터는 pandas.get_dummies()를 이용해 one-hot encoding 형식으로 변환
for col in categorical_feature:
    one_hot = pandas.get_dummies(sales[col], prefix=col)
    sales = pandas.concat([sales, one_hot], axis=1)
    sales.drop(col, axis=1, inplace=True)

#연속형 데이터의 모든 특성의 값이 동일한 중요도를 갖도록 하기 위해 MinMaxSclaer() 적용
mms = MinMaxScaler()
mms.fit(sales)
sales_after_preprocessing = mms.transform(sales)

#적당한 K값 계산 (SSD(Sum of Squared Distance)가 수렴하기 시작한 지점이 적당한 K라 볼 수 있음)
SSD = []
for k in range(1, 15):
    model = KMeans(n_clusters=k)
    model.fit(sales_after_preprocessing)
    SSD.append(model.inertia_)

#K값의 변화에 따른 SSD값 비교
matplotlib.pyplot.plot(range(1, 15), SSD, 'bx-')
matplotlib.pyplot.xlabel('K')
matplotlib.pyplot.ylabel('SSD')
matplotlib.pyplot.title('Finding Optimal K')
matplotlib.pyplot.show()
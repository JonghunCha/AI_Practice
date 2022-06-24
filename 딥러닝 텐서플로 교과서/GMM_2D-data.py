import numpy
import sys
import matplotlib.pyplot
from sklearn.mixture import GaussianMixture

numpy.set_printoptions(threshold=sys.maxsize)

#데이터 호출
data = numpy.load("./data.npy")
print(data.shape)
for i in range(10):
    print(data[i])

#GMM생성 및 데이터 적용
model = GaussianMixture(n_components=2)
model.fit(data)
print(model.means_)
print("\n")
print(model.covariances_)

#linspace는 -1과 6을 양끝으로 한 구간을 의미, meshgrid는 이러한 linspace 2개를 받아 x축 y축으로 사용함을 의미
x, y = numpy.meshgrid(numpy.linspace(-1, 6), numpy.linspace(-1, 6))
#numpy.ravel()은 데이터를 1차원으로 바꾸어 준다
xx = numpy.array([x.ravel(), y.ravel()]).T
##score_samples는 샘플에 대해 확률밀도함수(PDF)를 그려준다
z = model.score_samples(xx)
z = z.reshape((50, 50))

#contour는 등치선을 그려준다
matplotlib.pyplot.contour(x, y, z)
#scatter를 통해 학습에 사용된 data의 위치를 그려준다
matplotlib.pyplot.scatter(data[:,0], data[:,1])
matplotlib.pyplot.show()
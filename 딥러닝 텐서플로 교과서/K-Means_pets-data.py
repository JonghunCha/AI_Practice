"""
os : windows 10
gpu : RTX 3060
tensorflow : 2.4.0
python : 3.8
cuda : 11.0
cudnn : 8.0.5
anaconda : 4.13.0

-데이터-
47개의 개, 고양이 사진으로 이루어짐

-목적-
pre-train된 MobileNetV2(weight="imagenet")을 이용하여 각 이미지의 특성을 추출하고 해당 특성을 KMeans를 통하여 클러스터링 하는 것

-결과-
개와 고양이 클러스터링을 올바르게 하는 것을 파악함

또한, 이번 데이터의 경우 클래스의 수가 명확하지만 클래스의 수를 알 수 없을 때는 실루엣이나 엘보같은 방법을 이용하여 적정한 클러스터의 갯수를 파악할 수 있음
"""
import tensorflow
import numpy
import matplotlib.pyplot
import cv2
import os
import glob
import shutil
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#GPU설정
gpus = tensorflow.config.experimental.list_physical_devices("GPU")  #cuDNN초기화에 문제가 있을 경우 실행
if gpus:
    try:
        for gpu in gpus:    #GPU가 2개 이상일 경우 메모리를 균등하게 사용하도록 조정
            tensorflow.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tensorflow.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

#데이터셋 준비
input_dir = "./pets"
glob_dir = input_dir + "/*jpg"

images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
paths = [file for file in glob.glob(glob_dir)]
images = numpy.array(numpy.float32(images).reshape(len(images), -1) / 255)

#MobileNetV2를 통한 이미지의 특성 추출
mobilenet = tensorflow.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
predictions = mobilenet.predict(images.reshape(-1, 224, 224, 3))
pred_images = predictions.reshape(images.shape[0], -1)

#KMeans클러스터링 구성
K = 2
model = KMeans(n_clusters=K, random_state=10)
model.fit(pred_images)
KMeans_predictions = model.predict(pred_images)
shutil.rmtree("./output")
for i in range(K):
    os.makedirs("./output" + str(i))
for i in range(len(paths)):
    shutil.copy2(paths[i], "./output" + str(KMeans_predictions[i]))

#KMeans에서 클래스 개수 파악하기기(실루엣을 활용한 방법을 이용(엘보를 이용한 방법도 존재))
sil = []
kl = []
K_MAX = 10
for k in range(2, K_MAX + 1):
    KMeans_predictions = KMeans(n_clusters=k, random_state=10).fit(pred_images)
    labels = KMeans_predictions.labels_
    sil.append(silhouette_score(pred_images, labels, metric="euclidean"))
    kl.append(k)

#실루엣 값 시각화
matplotlib.pyplot.plot(kl, sil)
matplotlib.pyplot.ylabel("Silhoutte Score")
matplotlib.pyplot.show()
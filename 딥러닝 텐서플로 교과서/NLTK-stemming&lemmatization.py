"""
어간 추출(stemming)과 표제어 추출(lemmatizatino) 예제 코드입니다.
"""
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

#포터 알고리즘
stemmer = PorterStemmer()
print(stemmer.stem("obesses"), stemmer.stem("obssesed"))
print(stemmer.stem("standardizes"), stemmer.stem("standardization"))
print(stemmer.stem("national"), stemmer.stem("nation"))
print(stemmer.stem("absentness"), stemmer.stem("absently"))
print(stemmer.stem("tribalical"), stemmer.stem("tribalicalized"))
print("\n")

#랭커스터 알고리즘
stemmer = LancasterStemmer()
print(stemmer.stem("obesses"), stemmer.stem("obssesed"))
print(stemmer.stem("standardizes"), stemmer.stem("standardization"))
print(stemmer.stem("national"), stemmer.stem("nation"))
print(stemmer.stem("absentness"), stemmer.stem("absently"))
print(stemmer.stem("tribalical"), stemmer.stem("tribalicalized"))
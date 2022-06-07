"""
nltk를 이용해 불용어(stopwords)를 제거하는 예시 코드입니다
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#단어 단위로 토큰화
sample = "One of the first things that we ask ourselves is what are the pros and cons of any task we perform."
tokens = word_tokenize(sample)
print(tokens)
print("\n")

#불용어를 제거
tokens_without_stopwords = [word for word in tokens if not word in stopwords.words("english")]
print(tokens_without_stopwords)
print("\n")
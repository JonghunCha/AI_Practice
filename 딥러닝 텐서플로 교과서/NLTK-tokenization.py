"""
NLTK를 이용한 자연어 토큰화 예제 코드입니다
"""
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

#문장 단위 토큰화(마침표, 느낌표, 물음표 등 문장의 마지막을 뜻하는 기호에 따라 분리)
sample = "Firstly, the most basic methods are focusing on nouns only. " \
         "As keywords are usually nouns, many methods extract nouns using POS tagger and use it. " \
         "The strength of these methods is the ease of use and intuitiveness. " \
         "However, because they ignore other words except nouns, " \
         "it is impossible to extract keywords that are modified by adjectives or adverbs and multi-word keywords."
tokens = sent_tokenize(sample)
print(tokens)
print("\n")

#단어 단위 토큰화(띄어쓰기를 기준으로 문장을 구분-영어 기준)
sample = "This is about deep learning"
tokens = word_tokenize(sample)
print(tokens)
print("\n")

#어퍼스트로피가 포함된 문장에서 단어 토큰화
sample = "It's nothing that you don't already know except most people aren't aware of how their inner world works"
tokens = WordPunctTokenizer().tokenize(sample)
print(tokens)
print("\n")

#keras를 이용한 단어 토큰화
sample = "It's nothing that you don't already know except most people aren't aware of how their inner world works"
tokens = text_to_word_sequence(sample)
print(tokens)
print("\n")
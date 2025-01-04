import pickle
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.layers import Dense
import tensorflow as tf
from konlpy.tag import Okt
import numpy as np
import requests
import gensim
import pickle
import json
import nltk
import os

class InitModel(object):

    def __init__(self):
        print("초기화 시작")
        self.data = None
        self.tokens = None
        self.selected_words = None
        self.train_docs = None
        self.test_docs = None
        print("데이터 파일 읽기 시작")
        self.train_data = self.read_data('D:/python/police/ratings_train.txt')
        print("훈련 데이터 읽기 완료")
        self.test_data = self.read_data('D:/python/police/ratings_test.txt')
        print("테스트 데이터 읽기 완료")
        print("Okt 객체 생성 중...")
        self.okt = Okt()
        print("Okt 객체 생성 완료")
        self.make_tokens()

    def read_data(self, filename):
        print(f"{filename} 파일 읽기 시작")
        with open(filename, 'r', encoding='UTF8') as f:
            self.data = [line.split('\t') for line in f.read().splitlines()]
            # txt 파일의 헤더(id document label)는 제외하기
            self.data = self.data[1:]
        print(f"{filename} 파일 읽기 완료")
        return self.data

    def tokenize(self, doc):
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in self.okt.pos(doc, norm=True, stem=True)]

    def make_tokens(self):
        print("make_tokens 시작")

        # JSON 파일이 존재하는 경우, 이를 로드하여 사용
        if os.path.isfile('D:/python/police/train_docs.json') and os.path.isfile('D:/python/police/test_docs.json'):
            print("기존 JSON 파일 로드 중...")
            with open('D:/python/police/train_docs.json', encoding="utf-8") as f:
                self.train_docs = json.load(f)
            with open('D:/python/police/test_docs.json', encoding="utf-8") as f:
                self.test_docs = json.load(f)
        else:
            # JSON 파일이 없을 경우 전체 데이터를 토큰화하고 JSON 파일로 저장
            self.train_docs = [(self.tokenize(row[1]), row[2]) for row in self.train_data]
            self.test_docs = [(self.tokenize(row[1]), row[2]) for row in self.test_data]
            print("전체 데이터 토큰화 완료")

            # 토큰화된 데이터를 JSON 파일로 저장
            with open('D:/python/police/train_docs.json', 'w', encoding="utf-8") as make_file:
                json.dump(self.train_docs, make_file, ensure_ascii=False, indent="\t")
            with open('D:/python/police/test_docs.json', 'w', encoding="utf-8") as make_file:
                json.dump(self.test_docs, make_file, ensure_ascii=False, indent="\t")
            print("토큰화 데이터 JSON 파일 저장 완료")

        # 토큰 생성 및 샘플 출력
        self.tokens = [t for d in self.train_docs for t in d[0]]
        print("토큰 일부:", self.tokens[:5])

        # tokens.txt 파일로 토큰 저장
        with open('D:/python/police/tokens.txt', 'wb') as token_file:
            pickle.dump(self.tokens, token_file)
        print("tokens.txt 파일 저장 완료")

        self.text_common()

    def text_common(self):
        print("text_common 시작")
        text = nltk.Text(self.tokens, name='NMSC')
        self.selected_words = [f[0] for f in text.vocab().most_common(10000)]
        print("선택된 단어 수:", len(self.selected_words))
        self.make_data_set()

    def term_frequency(self, doc):
        return [doc.count(word) for word in self.selected_words]

    def make_data_set(self):
        print("데이터셋 생성 중...")
        train_x = [self.term_frequency(d) for d, _ in self.train_docs]
        test_x = [self.term_frequency(d) for d, _ in self.test_docs]
        train_y = [c for _, c in self.train_docs]
        test_y = [c for _, c in self.test_docs]
        print("데이터셋 생성 완료")
        self.model_learn(train_x, test_x, train_y, test_y)

    def model_learn(self, train_x, test_x, train_y, test_y):
        print("모델 학습 시작")

        x_train = np.asarray(train_x).astype('float32')
        x_test = np.asarray(test_x).astype('float32')
        y_train = np.asarray(train_y).astype('float32')
        y_test = np.asarray(test_y).astype('float32')

        # 모델 구성하기. selected_words의 길이에 맞게 input_shape 설정
        model = models.Sequential()
        model.add(Dense(64, kernel_initializer='uniform', activation='relu', input_shape=(len(self.selected_words),)))
        model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # 모델 학습과정 설정.
        model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                      loss=losses.binary_crossentropy,
                      metrics=[metrics.binary_accuracy])

        # 모델 학습하기.
        print("모델 훈련 시작")
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10,
                            batch_size=512)

        print("모델 평가 중...")
        results = model.evaluate(x_test, y_test)
        print("모델 평가 결과:", results)

        # 모델 저장.
        model.save('D:/python/police/emotional_analysis.h5')
        print("모델 저장 완료")


# 실행
if __name__ == "__main__":
    print("InitModel 실행")
    model = InitModel()
    print("InitModel 실행 완료")

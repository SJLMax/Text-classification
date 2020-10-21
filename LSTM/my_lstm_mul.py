# @Time : 2020/6/29 19:34 
# @Author : Shang
#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
# import matplotlib
# # import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from  sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#第三步  LSTM建模

#3.1
# 设置最频繁使用的50000个词(在texts_to_matrix是会取前MAX_NB_WORDS,会取前MAX_NB_WORDS列)

df=pd.read_excel(r'./alldata_lstm.xlsx')


for i in range(1):
    df_train = pd.read_excel(r'./train'+str(i)+'.xls')
    df_test = pd.read_excel(r'./test'+str(i)+'.xls')
    MAX_NB_WORDS = 50000
    # 每条cut_review最大的长度
    MAX_SEQUENCE_LENGTH = 300
    # 设置Embeddingceng层的维度
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['cut_review'].values)
    word_index = tokenizer.word_index
    print('共有 %s 个不相同的词语.' % len(word_index))  # 结果  69599

    # 3.2
    X = tokenizer.texts_to_sequences(df['cut_review'].values)  # 定义模型需要用到X
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    X_train = tokenizer.texts_to_sequences(df_train['cut_review'].values)
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)

    X_test = tokenizer.texts_to_sequences(df_test['cut_review'].values)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    Y_train = pd.get_dummies(df_train['cat_id']).values
    Y_test = pd.get_dummies(df_test['cat_id']).values

    # 3.3 划分训练集和测试集

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    # 3.4 定义模型
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # 3.5 训练模型
    epochs =10
    batch_size = 32
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    accr = model.evaluate(X_test, Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    # 3.6 结果评估
    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis=1)
    Y_test = Y_test.argmax(axis=1)
    print('accuracy %s' % accuracy_score(y_pred, Y_test))
    cat_id_df = df[['类别', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
    print(classification_report(Y_test, y_pred, digits=4, target_names=cat_id_df['类别'].values))


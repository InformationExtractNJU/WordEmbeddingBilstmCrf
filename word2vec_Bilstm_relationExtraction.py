from gensim.models import word2vec
import jieba
#!/usr/bin/env python
# coding: utf-8

import os
import codecs
import re
import random
import string
from tqdm import tqdm
import pandas as pd
import numpy as np
from zhon.hanzi import punctuation
from sklearn.model_selection import train_test_split
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_contrib.layers import CRF
import tensorflow as tf
import keras
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
tf.logging.set_verbosity(tf.logging.ERROR)

# 加载之前训练的word2vec_model模型
word2vec_model = word2vec.Word2Vec.load("word2vec_model/word2vec_word_embedding.model")
# print(word2vec_model[['汽车','吉利']])
def getWordEmbedding(word2vec_model,text,maxlen):
    word_embedding = word2vec_model[text].tolist()
    if(len(word_embedding) > maxlen):
        word_embedding = word_embedding[0:maxlen]
    else:
        word_0_vector = []
        for i in range(0,256):
            word_0_vector.append(0)
        startIndex = len(word_embedding)
        for i in range(startIndex,maxlen):
            word_embedding.append(word_0_vector)
    return  word_embedding

def get_traindata():
    reader = open('words_relation.txt', encoding='utf-8-sig')
    train_data = []
    list_data = reader.readlines()
    count = 0
    for i in range(len(list_data)):
        words_list = []
        relation_tag = []
        if i % 2 != 0:
            list_data[i] = list_data[i].replace('\n','')
            for j in list_data[i].split(' '):
                # print(j)
                relation_tag.append(int(j))
            words_list = list_data[i-1].replace('\n','').split(' ')
            train_data.append((count,words_list,relation_tag))
            count = count + 1
    return train_data
train_data = get_traindata()
print(train_data[0][0])
print(train_data[0][1])
print(train_data[0][2])
print(train_data[0])
print('数据加载完毕')
# 接下来定义模型
def makeBilstmCNNModel():
    x_in = keras.layers.Input(shape=(256, 256,), dtype='float32')
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))(x_in)
    dropout = keras.layers.Dropout(0.4)(lstm)
    dense1 = keras.layers.TimeDistributed(keras.layers.Dense(128, activation='relu'))(dropout)
    conv1d1 = keras.layers.Conv1D(64, 3, activation='relu')(dense1)
    conv1d2 = keras.layers.Conv1D(64, 3, activation='relu')(conv1d1)
    maxpooling1D = keras.layers.MaxPooling1D(3)(conv1d2)
    convld3 = keras.layers.Conv1D(64, 3, activation='relu')(maxpooling1D)
    convld4 = keras.layers.Conv1D(64, 3, activation='relu')(convld3)
    globalaveragepooling1d = keras.layers.GlobalAveragePooling1D()(convld4)
    # flatten = keras.layers.Flatten()(globalaveragepooling1d)
    out = keras.layers.Dense(12, activation='sigmoid')(globalaveragepooling1d)
    model = keras.models.Model(inputs=x_in, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
model = makeBilstmCNNModel()
# 采用5折交叉验证
#
kf = KFold(n_splits=5)
f1_news_score = []
recall_news_score = []
precision_news_score = []
iteration_count = 1
result_report = []
for train,test in kf.split(range(len(train_data))):
    save_path = 'model/model'
    save_path += str(iteration_count)
    filepath = "model_{epoch:02d}.hdf5"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        keras.callbacks.ModelCheckpoint(os.path.join(save_path, filepath),
                                        monitor='val_loss', save_best_only=True, verbose=0),
    ]
    print("这是第" + str(iteration_count) + '轮交叉验证')
    iteration_count = iteration_count + 1
    id_train, text_id_train, X_train, Y_train = [], [], [], []
    id_test, text_id_test, X_test,Y_test = [], [], [], []
    maxlen = 256
    # 对训练集进行处理
    id_count = 0
    for i in train:
        d = train_data[i]
        text = d[1][:maxlen]
        y = d[2][:maxlen]
        x = getWordEmbedding(word2vec_model,text,maxlen)
        X_train.append(x)
        Y_train.append(y)
        id_train.append(id_count)
        id_count = id_count + 1
        text_id_train.append([d[0]])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 对测试集进行处理
    id_count = 0
    for i in test:
        d = train_data[i]
        text = d[1][:maxlen]
        y = d[2][:maxlen]
        x = getWordEmbedding(word2vec_model,text,maxlen)
        X_test.append(x)
        Y_test.append(y)
        id_test.append([id_count])
        id_count = id_count + 1
        text_id_test.append([d[0]])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    # print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
    # 进行训练
    history = model.fit(X_train,Y_train,batch_size=64,epochs=20,
                        validation_data=(X_test,Y_test), verbose=1, callbacks=callbacks)
    # 显示训练信息
    hist = pd.DataFrame(history.history)
    # 进行预测
    pred = model.predict(X_test,verbose=1)
    pred_labels = pred.tolist()
    test_labels = Y_test.tolist()
    # print(test_labels)
    # print(pred_labels)
    # 将结果抓换成字符串
    pred_labels_str = []
    test_labels_str = []
    for i in range(len(pred_labels)):
        str_temp_pred = ''
        str_temp_test = ''
        for j in range(len(pred_labels[i])):
            str_temp_pred += str(pred_labels[i][j])
            str_temp_pred += ' '
            str_temp_test += str(test_labels[i][j])
            str_temp_test += ' '
        pred_labels_str.append(str_temp_pred)
        test_labels_str.append(str_temp_test)
    # print(test_labels_str)
    # print(pred_labels_str)
    f1_news_score.append(f1_score(test_labels_str,pred_labels_str))
    precision_news_score.append(precision_score(test_labels_str,pred_labels_str))
    recall_news_score.append(recall_score(test_labels_str,pred_labels_str))
    # 统计相关信息
    result_report.append(classification_report(test_labels_str,pred_labels_str))
    result_str = ''
    for i in range(len(pred_labels_str)):
        result_str += pred_labels_str[i]
        result_str += '\n'
    write_txt = open('Test_case/Fold' + str(iteration_count - 1) + 'Result.txt','w',encoding='utf-8')
    write_txt.writelines(result_str)
    write_txt.close()
print('平均f1值')
print(np.array(f1_news_score).mean())
print('平均recall')
print(np.array(recall_news_score).mean())
print('平均precision')
print(np.array(precision_news_score).mean())
resultScore_write = open('resultScore.txt','w',encoding='utf-8')
resultScore_write.writelines(result_report)
resultScore_write.close()
print('写入完成')
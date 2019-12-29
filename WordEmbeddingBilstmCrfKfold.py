#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from gensim.models import word2vec
from keras.models import Sequential
import jieba
from collections import Counter

tf.logging.set_verbosity(tf.logging.ERROR)


# 标签统计
tags = ['O', 'B-COMPANY', 'I-COMPANY', 'B-TEL', 'I-TEL', 'B-CAR', 'I-CAR', 'B-HARDWARE', 'I-HARDWARE', 'B-PATENT', 'I-PATENT', 'B-SOFTWARE', 'I-SOFTWARE', 'B-PER', 'I-PER', 'B-SERVICE', 'I-SERVICE', 'B-TIME', 'I-TIME', 'B-LOC', 'I-LOC']
tag2idx = {tag:i+1 for i, tag in enumerate(tags)}
tag2idx['-PAD-'] = 0
n_tags = len(tag2idx)
tag2idx

print (tag2idx)

import re
def get_train_data():
    train_data = []
    n2id = []
    count = 0
    reader = open('../train_data/sentences.txt',encoding = 'utf-8-sig')
    list_data = reader.readlines()
    for i,element in enumerate(list_data):
        if i % 2 != 0:
            n2id = []
            tags_str = list_data[i]
            text_str = list_data[i-1]
            text_str = text_str.replace(' ','')
            text_str = text_str.replace('\n','')
            tags_str.replace('\n',' ')
            tags_str_list = tags_str.split(' ')
            for j,e in enumerate(tags_str_list):
                # print(e)
                list_temp = []
                if e != '\n' and e != '':
                    list_temp.append(tag2idx[e])
                    n2id.append(list_temp)
            train_data.append((count,text_str,n2id))
            count = count+1
    return train_data

train_data = get_train_data()
print('数据读取完毕')
print (len(train_data))
print(len(train_data[0]))

print(train_data[0])


# 定义模型
x1_in = keras.layers.Input(shape=(None,None,))
x2_in = keras.layers.Input(shape=(None,))

print (x1_in)
print (x1_in.shape)


# 加载预训练的word2vec模型
word2vec_model = word2vec.Word2Vec.load("word2vec_model/word2vec_word_embedding.model")

# word2vec_output=word2vec_model[x1_in]

# lstm = keras.layers.Bidirectional(keras.layers.LSTM(units = 128,return_sequences = True))(x1_in )
# drop = keras.layers.Dropout(0.4)(lstm)
# dense = keras.layers.TimeDistributed(keras.layers.Dense(128,activation='relu'))(drop)
# crf = CRF(n_tags)
# out = crf(dense)
# model = keras.models.Model(inputs=[x1_in],outputs=out)

model = Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units = 128, return_sequences=True)))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(128,activation='relu')))
crf=CRF(n_tags)
model.add(crf)
model.compile(loss=crf.loss_function,optimizer='adam',metrics=[crf.accuracy])
# model.summary()

save_path = 'model'
filepath="model_{epoch:02d}-{val_crf_viterbi_accuracy:.4f}.hdf5"

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,verbose=0),
    keras.callbacks.ModelCheckpoint(os.path.join(save_path,filepath),
monitor='val_loss',save_best_only=True, verbose= 0),
]

kf = KFold(n_splits=5)
id_,text_id,X1,X2,Y=[],[],[],[],[]
f1_news_score = []
recall_news_score = []
precision_news_score = []
iteration_count = 1

def changetoThreeDim(two_dim):
    res = []
    for word in two_dim:
        word_vec = []
        for j in word:
            dim_vec = []
            dim_vec.append(j)
            word_vec.append(dim_vec)
        res.append(word_vec)
    return res

def getWord2vec(words):
    vec=[]
    for i in words:
        try:
            temp=word2vec_model[i]
            # print (temp)
            # print (len(temp))
            # print (len(i))
            vec.append(temp)
        except:
            vec.append([[0]*256]*len(i))
            # print ("not found")
    # print (vec)
    return vec

# 进行分词
def cut_voc(para):
    seg_list = jieba.cut(para)
    sen_cut = []
    for w in seg_list:
        sen_cut.append(w)
    return sen_cut

# [1:11]
for train,test in kf.split(range(len(train_data))):
    print("这是第"+str(iteration_count)+'轮交叉验证')
    iteration_count = iteration_count+1
    id_train, text_id_train, X1_train, X2_train, Y_train = [], [], [], [], []
    id_test, text_id_test, X1_test, X2_test, Y_test = [], [], [], [], []
    maxlen = 256
    # 对训练集进行处理
    id_count = 0
    for i in train:
        d = train_data[i]
        text = d[1][:maxlen]
        y = d[2][:maxlen]
        # x1, x2 = tokenizer.encode(first=text)
        # print (text)
        # words = []
        # for word in text:
        #     words.append(word)
        words=cut_voc(text)
        y_after_cut_word = []
        index=0
        for i in words:
            flag_findB=False
            try:
                for j in range(index, index + len(i)):
                    if (y[j][0] != 0 and tags[y[j][0] - 1][0:1] == 'B'):
                        # print (tags[y[j][0] - 1])
                        flag_findB = True
                        y_after_cut_word.append(y[j])
                        break
                if (flag_findB == False):
                    label_list = []
                    for k in range(index, index + len(i)):
                        label_list += y[k]
                    label_index_count = Counter(label_list)
                    top_one_label = label_index_count.most_common(1)[0][0]
                    # print (top_one_label)
                    y_after_cut_word.append([top_one_label])
                index += len(i)
            except:
                print ("E",end="")
        # print ("-------------------------------")
        # print (words)
        # print (y_after_cut_word)

        X1_train.append(words)
        Y_train.append(y_after_cut_word)
        id_train.append(id_count)
        id_count = id_count+1
        text_id_train.append([d[0]])
    # X1_train = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_train,padding="post", value=0)
    # X2_train = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_train, padding="post", value=0)
    Y_train = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=Y_train, padding="post", value=0)
    Y_train = [keras.preprocessing.utils.to_categorical(i, num_classes=n_tags) for i in Y_train]
    # print(len(X1_train))
    #
    # print (len(X1_train),len(Y_train))
    X1_train = getWord2vec( X1_train)
    # print ("----------------------------***************")
    # print (len(X1_train))
    # print (len(Y_train))
    # print (len(X1_train))
    # print (Y_train[0])
    X1_train = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_train, padding="post", value=0)
    # print (len(X1_train), len(Y_train))


    # 对测试集进行处理
    id_count = 0
    for i in test:
        d = train_data[i]
        text = d[1][:maxlen]
        y = d[2][:maxlen]
        # x1, x2 = tokenizer.encode(first=text)
        # X1_test.append(x1)
        # X2_test.append(X2)
        words = cut_voc(text)
        y_after_cut_word = []
        index = 0
        for i in words:
            flag_findB = False
            try:
                for j in range(index, index + len(i)):
                    if (y[j][0] != 0 and tags[y[j][0] - 1][0:1] == 'B'):
                        # print (tags[y[j][0] - 1])
                        flag_findB = True
                        y_after_cut_word.append(y[j])
                        break
                if (flag_findB == False):
                    label_list = []
                    for k in range(index, index + len(i)):
                        label_list += y[k]
                    label_index_count = Counter(label_list)
                    top_one_label = label_index_count.most_common(1)[0][0]
                    # print (top_one_label)
                    y_after_cut_word.append([top_one_label])
                index += len(i)
            except:
                print ("E",end="")
        # print ("-------------------------------")
        # print (words)
        # print (y_after_cut_word)

        X1_test.append(words)
        Y_test.append(y_after_cut_word)
        id_test.append([id_count])
        id_count = id_count + 1
        text_id_test.append([d[0]])
    # X1_test = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_test, padding="post", value=0)
    # X2_test = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_test, padding="post", value=0)
    Y_test = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=Y_test, padding="post", value=0)
    Y_test = [keras.preprocessing.utils.to_categorical(i, num_classes=n_tags) for i in Y_test]
    print(len(id_train), len(id_test), len(text_id_train), len(text_id_test), len(X1_train), len(X1_test),
        len(Y_train), len(Y_test))
    # 进行训练
    X1_test = getWord2vec(X1_test)
    X1_test = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_test, padding="post", value=0)

    history = model.fit(X1_train, np.array(Y_train), batch_size=32, epochs=20,
                        validation_data=(X1_test, np.array(Y_test)), verbose=1, callbacks=callbacks)
    # 显示训练信息
    hist = pd.DataFrame(history.history)
    print(hist.head())
    # 进行预测
    test_pred = model.predict(X1_test, verbose=1)
    print(test_pred)
    print(test_pred.shape)
    # 定义结果标签
    idx2tag = {i: w for w, i in tag2idx.items()}
    print('tag2idx:', tag2idx)
    print('idx2tag:', idx2tag)

    # 转换实体的预测标签函数
    def pred2label(pred):
        out = []
        for pred_i in pred:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p)
                out_i.append(idx2tag[p_i].replace("-PAD-", "O"))
            out.append(out_i)
        return out

    pred_labels = pred2label(test_pred)
    test_labels = pred2label(Y_test)

    # 查看相应的F1值
    print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
    print("F1-score: {:.1%}".format(precision_score(test_labels, pred_labels)))
    print("F1-score: {:.1%}".format(recall_score(test_labels, pred_labels)))
    f1_news_score.append(f1_score(test_labels, pred_labels))
    precision_news_score.append(precision_score(test_labels, pred_labels))
    recall_news_score.append(recall_score(test_labels, pred_labels))
    # 统计相关信息
    print(classification_report(test_labels, pred_labels))

    # 随机抽样
    sample_id = random.sample(range(len(id_test)), 1)[0]
    sample_X1 = X1_test[sample_id]
    # sample_X2 = X2_test[sample_id]
    tid = id_test[sample_id][0]
    sample_text_id = text_id_test[sample_id]
    print(sample_text_id)
    sample_data = train_data[tid]
    print(sample_data)
    sample_Y = Y_test[sample_id]
    print(sample_Y)
    print ("------------------------sample_X1")
    print (sample_X1)
    sample_X_input=np.array([sample_X1])
    # sample_X_input.append(sample_X1)
    print (sample_X_input)
    print (len(sample_X_input))
    # print (sample_X_input.shape)
    predict = model.predict(sample_X_input)
    print(predict.shape)
    pred = np.argmax(predict, axis=-1).reshape([-1])
    true = np.argmax(sample_Y, axis=-1)

    pred_label = [idx2tag[i] for i in pred]
    true_label = [idx2tag[i] for i in true]

    for c, t, p in zip(sample_data[1], pred_label, true_label):
        if t != "-PAD-":
            print("{:15}: {:5} {}".format(c, t, p))
print('平均f1值')
print(np.array(f1_news_score).mean())
print('平均recall')
print(np.array(recall_news_score).mean())
print('平均precision')
print(np.array(precision_news_score).mean())
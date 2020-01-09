from gensim.models import word2vec
import jieba
import os
import os.path
# word2vec.Text8Corpus('NewsCar_new_after_process/1/2.txt')

# # 加载分句后的文件
# train_data = []
# for i in range(2015,2020):
#     for j in range(1,13):
#         filedir = 'sentences'+'/'+'sentences'+'/'+str(i)+'/'+str(j)
#         filenames = os.listdir(filedir)
#         for filename in filenames:
#             filepath = filedir+'/'+filename
#             reader = open(filepath,encoding = 'utf-8-sig')
#             list_txt = reader.readlines()
#             for line in list_txt:
#                 train_data.append(line)
# print(len(train_data))
reader = open('sentences_relation.txt',encoding = 'utf-8-sig')
train_data = reader.readlines()
print('数据读取完毕')
def cut_voc(para):
    seg_list = jieba.cut(para)
    # for w in seg_list:
    #     print (w.word, w.flag)
    return seg_list


words = []
relation_tag = []
for i in range(len(train_data)):
    if(i %2 == 0):
        sen = train_data[i]
        sen = sen.replace(' ','')
        sen = sen.replace('\n', '')
        # print (sen)
        sen=cut_voc(sen)
        # print (sen)
        sen_cut=[]
        for w in sen:
            sen_cut.append(w)
        words.append(sen_cut)
    else:
        relation_tag.append(train_data[i])
        # print (words)
#
# print (len(words))
# print(words)
# print (words[0])
model = word2vec.Word2Vec(words, size=256, min_count=1)
print(words[0])
# size 表示向量维度 min_count表示最小出现次数
# model = word2vec.Word2Vec.load("word2vec_model/word2vec_word_embedding.model")
#
# # 计算和车最相似的5个字
# x=model.most_similar("技术",topn=5)
# print (x)
# #
# simliar=model.similarity('汽车','能源')
# print (simliar)
# # 输出'汽车'的词向量
# print(model[['汽车']])
#
# two_dim=model[['汽','车']]
# res=[]
# for word in two_dim:
#     word_vec=[]
#     for j in word:
#         dim_vec=[]
#         dim_vec.append(j)
#         word_vec.append(dim_vec)
#     res.append(word_vec)
#
# print (res)
#
# vocab=model.wv.vocab
# for word in vocab:
#     print (word)

# 保存模型
model.save("word2vec_model/word2vec_word_embedding.model")
# 对应的加载方式
model_2 = word2vec.Word2Vec.load("word2vec_model/word2vec_word_embedding.model")
text = ['汽车','吉利']
word_embedding = model_2[text]
print(word_embedding.shape)
f = open('words_relation.txt','w',encoding='utf-8')
for i in range(len(words)):
    for j in range(len(words[i])):
        if j != len(words[i])-1:
            f.writelines(words[i][j])
            f.writelines(' ')
        else:
            f.writelines(words[i][j])
    f.writelines('\n')
    f.writelines(relation_tag[i][0:len(relation_tag[i])-2])
    f.writelines('\n')
f.close()




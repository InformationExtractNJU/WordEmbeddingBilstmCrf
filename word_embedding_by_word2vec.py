from gensim.models import word2vec
import jieba
# word2vec.Text8Corpus('NewsCar_new_after_process/1/2.txt')

# 加载分句后的文件
# reader = open('../train_data/sentences_relation.txt',encoding = 'utf-8-sig')
# train_data = reader.readlines()
#
# def cut_voc(para):
#     seg_list = jieba.cut(para)
#     # for w in seg_list:
#     #     print (w.word, w.flag)
#     return seg_list
#
# print (train_data[0])
#
# words = []
# for i in range(len(train_data)):
#     if(i%2 == 0):
#         sen = train_data[i]
#         sen = sen.replace(' ','')
#         sen = sen.replace('\n', '')
#         # print (sen)
#         sen=cut_voc(sen)
#         # print (sen)
#         sen_cut=[]
#         for w in sen:
#             sen_cut.append(w)
#         words.append(sen_cut)
#         # print (words)
# #
# # print (len(words))
# print(words)
# print (words[0])
# model = word2vec.Word2Vec(words, size=256, min_count=1)

# size 表示向量维度 min_count表示最小出现次数
model = word2vec.Word2Vec.load("word2vec_model/word2vec_word_embedding.model")
#
# # 计算和车最相似的5个字
x=model.most_similar("技术",topn=5)
print (x)
#
simliar=model.similarity('汽车','能源')
print (simliar)
# 输出'汽车'的词向量
print(model[['汽车']])
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

# vocab=model.wv.vocab
# for word in vocab:
#     print (word)
#
# # 保存模型
# model.save("word2vec_model/word2vec_word_embedding.model")
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")




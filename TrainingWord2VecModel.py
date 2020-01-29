from gensim.models import word2vec
import jieba
import os
import os.path

class TrainingWord2VecModel:
    def cut_voc(self, para):
        seg_list = jieba.cut(para)
        return seg_list

    def trainingModel(self, filePath,modelOutputPath):
        """the input file must be like this:
            one line Chinese sentence with one line other sentence
        """
        reader = open(filePath, encoding='utf-8-sig')
        train_data = reader.readlines()
        words = []
        relation_tag = []
        for i in range(len(train_data)):
            if (i % 2 == 0):
                sen = train_data[i]
                sen = sen.replace(' ', '')
                sen = sen.replace('\n', '')
                sen = cut_voc(sen)
                sen_cut = []
                for w in sen:
                    sen_cut.append(w)
                words.append(sen_cut)
            else:
                relation_tag.append(train_data[i])
        model = word2vec.Word2Vec(words, size=256, min_count=1)
        model.save(modelOutputPath)

    def loadModel(self,modelPath):
        model = word2vec.Word2Vec.load(modelPath)
        return model




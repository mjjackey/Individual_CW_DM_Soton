import os
import sys
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
import string
import pprint
import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.externals import joblib ####### this is deprecated
import joblib
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from scipy.cluster.hierarchy import ward, dendrogram

import matplotlib.pyplot as plt

import nltk
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

from wordcloud import WordCloud,ImageColorGenerator, STOPWORDS
import collections
import seaborn as sns

from sklearn.decomposition import NMF, LatentDirichletAllocation

from nltk import pos_tag
import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

from sklearn.metrics import silhouette_score

MDS()

'''
全局变量和全局函数
'''
titles = ['DICTIONARY OF GREEK AND ROMAN GEOGRAPHY VOL. II',
          'THE WORKS OF CORNELIUS TACITUS VOL. V',
          'THE HISTORY OF THE PELOPONNESIA WAR VOL. II',
          'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE VOL. I',
          'THE HISTORY OF ROME VOL. I',
          'THE WHOLE GENUINE WORKS OF FLAVIUS JOSEPHUS VOL. II',
          'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE VOL. II',
          'THE DESCRIPTION OF GREECE VOL. M',
          'THE HISTORY OF ROME VOL. III',
          'LIVY VOL. III',
          'THE HISTORY OF THE PELOPONNESIA WAR VOL. I',
          'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE VOL. IV',
          'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE VOL. V',
          'THE HISTORICAL ANNALS OF CORNELIUS TACITUS VOL. I',
          'TITUS LIVIUS ROMAN',
          'THE WORKS OF JOSEPHUS WITH A LIFE VOL. IV',
          'THE WORKS OF CORNELIUS TACITUS VOL. IV',
          'LIVY VOL. V',
          'THE LIFE OF JOSEPHUS',
          'THE FIRST AND THIRTY-THIRD BOOKS OF PLINY\'S NATURAL HISTORY',
          'THE HISTORY OF ROME VOL. V',
          'THE HISTORIES OF CAIUS CORNELIUS TACITUS',
          'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE VOL. VI',
          'THE WORKS OF JOSEPHUS VOL. III']

n_samples = 24
n_features = 15000
n_features2= 15000
n_components = 10  # the number of clusters
n_top_words = 10

'''
存放每本书的类
'''
class AJBaseTextObject(object):

    def __init__(self, title,  raw_text=''):
        self.__title = title
        self.__raw_texts = raw_text

    @property
    def texts(self):
        return self.__raw_texts

'''
对所有书做统计的类
'''
class ATTextClassification(object):
    """
    类内部使用，不定义属性也可以
    """
    @property
    def tf_idf_matrix(self):
        return self.__tf_idf_matrix

    @tf_idf_matrix.setter
    def tf_idf_matrix(self, s):
        self.__tf_idf_matrix = s

    @property
    def all_books_texts(self):
        return self.__all_books_texts

    @all_books_texts.setter
    def all_books_texts(self,s):
        self.__all_books_texts = s

    @property
    def doc_list(self):
        return self.__doc_list

    @property
    def tfidf_feature_names(self):
        return self.__tfidf_feature_names

    @tfidf_feature_names.setter
    def tfidf_feature_names(self,s):
        self.__tfidf_feature_names=s

    def __init__(self):
        self.__all_books_texts = ""  # merge all books' texts
        self.__doc_list=[]   # the list of the all books' texts
        self.__books_list = []  # the list of all books

        self.__tf_idf_matrix = None
        self.__dist_matrix = None
        self.__tfidf_feature_names=None

    def add_book(self, book):
        self.__books_list.append(book)

    def print_word_count(self):
        _words = []
        for book in self.__books_list:
            text_arr = book.texts.strip().split()
            count = 0
            for w in text_arr:
                if len(w) > 0:
                    count += 1
            _words.append(count)

        print("wordcounts:",_words)

    def read_raw_text_from_path(self, data_path):
        # print('read data from {}'.format(data_path))
        with open(data_path) as data_file:
            data = data_file.read()
            #print len(data)
            data_file.close()
        return data

    def run(self):
        cwd = os.getcwd()
        fd_str = os.path.join(cwd, "clean_output3")
        filenames = []
        for name in os.listdir(fd_str):
            if name.endswith('.txt') and os.path.isfile(os.path.join(fd_str, name)):
                filenames.append(name)

                file_path = os.path.join(fd_str, name)
                text =self.read_raw_text_from_path(file_path)
                self.__doc_list.append(text)
                self.all_books_texts = self.all_books_texts+text+' '
                book = AJBaseTextObject(name, raw_text=text)
                self.add_book(book)

        # for name in filenames:
        #     print(name)

        self.print_word_count()

    def draw_wordbarchart(self):
        words_list = [word.strip() for text in self.doc_list for word in text.split()]
        print("words_list",words_list[:10])
        word_frequency = collections.Counter(words_list).most_common(250)
        print("word_frequency",type(word_frequency), word_frequency[:30])

        word_frequency_frame = pd.DataFrame(word_frequency[:30], columns=['words', 'frequency'])
        print("word_frequency_frame:\n",word_frequency_frame.head(5))
        sns.barplot(word_frequency_frame['frequency'], word_frequency_frame['words'])
        plt.title("Words Frequency in All Documents", fontsize=15)
        plt.xlabel('Words Frequency', fontsize=12)
        plt.ylabel('Words', fontsize=12)
        plt.savefig('./word_frequency.pdf', format='pdf')
        plt.show()


    def draw_wordcloud(self):
        word_cloud = WordCloud(
            max_words=250,  # 设置最大实现的字数
            stopwords=STOPWORDS,  # 设置停用词
            # font_path='simfang.ttf',  # 设置字体格式，如不设置显示不了中文
            max_font_size=40,  # 设置字体最大值
            random_state=300,  # 设置有多少种随机生成状态，即有多少种配色方案
            scale=5,
            width=500,
            height=300
        ).generate(self.all_books_texts)
        # word_cloud.to_file('./wordcloud.pdf')
        word_clord_array=word_cloud.to_array() #### 每个点的RGB像素数组
        print("word_cloud_array.shape",word_clord_array.shape)

        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('./word_cloud.pdf', format='pdf')
        plt.show()


    def tf_idf(self):
        files = os.listdir('./')
        saved_file_name = 'doc_matrix.pkl'
        # if saved_file_name not in files:

        #define vectorizer parameters
        # min_df=10 --- ignore terms that appear lesser than 10 times
        # max_features=None  --- Create as many words as present in the text corpus
        # changing max_features to 10k for memmory issues
        # analyzer='word'  --- Create features from words (alternatively char can also be used)
        # ngram_range=(1,1)  --- Use only one word at a time (unigrams)
        # strip_accents='unicode' -- removes accents
        # use_idf=1,smooth_idf=1 --- enable IDF
        # sublinear_tf=1   --- Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)

        # tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=250000,
        #                                  min_df=0.01, stop_words='english',
        #                                  use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 1))

        # tfidf_vectorizer = TfidfVectorizer(min_df=10, max_features=250000,
        #                                    strip_accents='unicode', analyzer='word', ngram_range=(1, 1),
        #                                    use_idf=1, smooth_idf=1, sublinear_tf=1,
        #                                    stop_words='english')

        #### Unigram
        tfidf_vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.9, max_features=n_features,  #When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold
                                           # strip_accents='unicode',
                                           analyzer='word', ngram_range=(1, 1),
                                           use_idf=1, smooth_idf=1, sublinear_tf=1,
                                           stop_words='english')

        self.tf_idf_matrix = tfidf_vectorizer.fit_transform(self.doc_list)  # fit the vectorizer to synopses
        print("tf_idf_matrix1:", self.tf_idf_matrix.shape)
        print(self.tf_idf_matrix[0][0])

        tf_idf_matrix_dense = self.tf_idf_matrix.todense()
        print("tf_idf_matrix_dense1: ", tf_idf_matrix_dense.shape)

        tfidf_feature = tfidf_vectorizer.get_feature_names()
        self.tfidf_feature_names=tfidf_feature
        print('tfidf_feature_names:' + str(len(tfidf_feature)))
        print('tfidf_feature_names top 10:', tfidf_feature[:10])
        print('tfidf_feature_names bottom 10:', tfidf_feature[:-10:-1])
        # print("tfidf vocabulary:", tfidf_vectorizer.vocabulary_)
        # print(list(tfidf_vectorizer.vocabulary_.values()))
        # print(np.array(list(tfidf_vectorizer.vocabulary_.values())))
        topn_index_list = np.array(list(tfidf_vectorizer.vocabulary_.values())).argsort()[::-1][:30] ######## 这边加新东西
        # print(topn_index_list)
        # topn_voc_list=[tfidf_vectorizer.vocabulary_.items()[i] for i in topn_index_list]
        # print("Top n tfidf vocabulary:",topn_voc_list)

        joblib.dump(self.tf_idf_matrix, saved_file_name)  # save

        # fig,ax = plt.subplots()
        # ax.imshow(tf_idf_matrix_dense.transpose()[:30, :])
        # # ax.imshow(self.tf_idf_matrix.transpose().todense()[:30,:])
        # ax.set_yticks(np.arange(30))
        # ax.set_yticklabels(self.tfidf_feature_names[:30])

        plt.imshow(tf_idf_matrix_dense.transpose()[:30, :])
        plt.yticks(np.arange(30),self.tfidf_feature_names[:30])
        plt.colorbar()
        plt.title('The 2D regular raster of TF.IDF matrix')
        plt.xlabel('24 Documents')
        plt.ylabel('The first 30 features')
        plt.savefig('./tfidf_matrix_imshow.pdf', format='pdf',bbox_inches='tight')
        plt.show()

        #####Bigrams
        tfidf_vectorizer2 = TfidfVectorizer(min_df=0.01, max_df=0.9, max_features=n_features,
                                           # strip_accents='unicode',
                                           analyzer='word', ngram_range=(2, 2),
                                           use_idf=1, smooth_idf=1, sublinear_tf=1,
                                           stop_words='english')

        tf_idf_matrix2 = tfidf_vectorizer2.fit_transform(self.doc_list)  # fit the vectorizer to synopses
        print("tf_idf_matrix2:", tf_idf_matrix2.shape)

        tf_idf_matrix_dense2= tf_idf_matrix2.todense()
        print("tf_idf_matrix_dense2: ", tf_idf_matrix_dense2.shape)

        # else:
        #         #     tfidf_matrix = joblib.load(saved_file_name)

        tfidf_feature2 = tfidf_vectorizer2.get_feature_names()
        print('tfidf_feature_names:' + str(len(tfidf_feature2)))
        print('tfidf_feature_names top 10:',tfidf_feature2[:10])
        print('tfidf_feature_names bottom 10:',tfidf_feature2[:-10:-1])

        # print("tfidf vocabulary:",tfidf_vectorizer2.vocabulary_)

        self.__dist_matrix = 1 - cosine_similarity(self.tf_idf_matrix) #24*24
        print("dist_matrix:",self.__dist_matrix.shape)


    def pos_tag(self):
        sent = pos_tag(self.tfidf_feature_names[:30])
        pattern = 'NP: {<DT>?<JJ>*<NN>}'  ######
        cp = nltk.RegexpParser(pattern)
        cs = cp.parse(sent)
        print("pos_tag of tfidf feature names",cs)

    def named_entity(self):
        ne=nltk.ne_chunk(self.tfidf_feature_names[:30])
        print("named_entity of tfidf feature names",ne)

    def lsa(self):
        pass

    '''
    use CountVectorizer
    '''
    def lda(self):
        tf_vectorizer = CountVectorizer(min_df=0.01, max_df=0.9, max_features=n_features2,
                                        strip_accents='unicode', analyzer="word", \
                                        ngram_range=(1, 1), \
                                        stop_words='english')
        tf = tf_vectorizer.fit_transform(self.doc_list)

        print("Fitting LDA models with tf features, "
              "n_samples=%d and n_features=%d..."
              % (n_samples, n_features2))
        lda = LatentDirichletAllocation(n_components=n_components, #Number of topics,default=10
                                        max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit(tf)
        tf_feature_names = tf_vectorizer.get_feature_names()
        # print("tf_feature_names:"+str(len(tf_feature_names)))
        # print("tf_feature_names bottom 10:",tf_feature_names[:-10:-1])
        print("Topics in LDA model:")
        self.print_top_words(lda, tf_feature_names, n_top_words)

    '''
    use TfidfVectorizer
    '''
    def nmf(self):
        tfidf_vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.9, max_features=n_features2,
                                           strip_accents='unicode', analyzer='word', ngram_range=(1, 1),
                                           use_idf=1, smooth_idf=1, sublinear_tf=1,
                                           stop_words='english')
        tfidf_matrix_nmf = tfidf_vectorizer.fit_transform(self.doc_list)

        # Fit the NMF model(Frobenius norm)
        print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
              "n_samples=%d and n_features=%d..."
              % (n_samples, n_features2))
        nmf = NMF(n_components=n_components, random_state=1, #Number of components, if n_components is not set all features are kept.
                  alpha=.1, l1_ratio=.5).fit(tfidf_matrix_nmf)

        print("Topics in NMF model (Frobenius norm):")
        self.tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        self.print_top_words(nmf, self.tfidf_feature_names, n_top_words)

        # Fit the NMF model(Kullback-Leibler divergence)
        print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
              "tf-idf features, n_samples=%d and n_features=%d..."
              % (n_samples, n_features2))
        nmf = NMF(n_components=n_components, random_state=1,
                  beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
                  l1_ratio=.5).fit(tfidf_matrix_nmf)

        print("Topics in NMF model (generalized Kullback-Leibler divergence):")
        self.print_top_words(nmf, self.tfidf_feature_names, n_top_words)

    '''
    Get words associated with each 'Topic'
    '''
    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            # print("topic:",len(topic))
            topic_feature_names_list=[feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            message = "Topic #%d: " % topic_idx
            message += " ".join(topic_feature_names_list)  # topic.argsort() 返回topic升序排列后的索引
            # 每一个topic都含有feature_names同容量的个数的元素, [:-n_top_words - 1:-1]切片，从索引的最后一个值到倒数第n_top_words个值
            print(message)
        print()


    def k_mean_clustering(self, n_clusters=5):
        num_clusters = n_clusters
        km = KMeans(n_clusters=num_clusters)
        km.fit(self.tf_idf_matrix)
        clusters = km.labels_.tolist()
        print('clusters:',clusters)

        # silhouette_avg=silhouette_score(self.tf_idf_matrix.todense(),clusters)
        silhouette_avg = silhouette_score(self.tf_idf_matrix, clusters)
        print("The silhouette_score is",silhouette_avg)

        books = {'title': titles, 'cluster': clusters}  # two columns
        frame = pd.DataFrame(books, index = [clusters] , columns = ['title', 'cluster'])
        # print(frame['cluster'].value_counts())
        print(frame.head(5))
        #grouped = frame['title'].groupby(frame['cluster']) #groupby cluster for aggregation purposes

        #print grouped.mean() #average rank (1 to 100) per cluster

        # convert two components as we're plotting points in a two-dimensional plane
        # "precomputed" because we provide a distance matrix
        # we will also specify `random_state` so the plot is reproducible.
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        pos = mds.fit_transform(self.__dist_matrix)  # shape (n_samples,n_components) 24*2
        xs, ys = pos[:, 0], pos[:, 1]  # the 1st column, the 2nd column
        # print(xs)
        # print(ys)

        #set up colors per clusters using a dict
        cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e',5:'black'}

        #set up cluster names using a dict
        cluster_names = {0: 'Group 1',
                         1: 'Group 2',
                         2: 'Group 3',
                         3: 'Group 4',
                         4: 'Group 5',
                         5: 'Group 6',
                         6: 'Group 7',
        }

        #create data frame that has the result of the MDS plus the cluster numbers and titles
        df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) # 4 columns

        #group by cluster
        groups = df.groupby('label')

        # set up plot
        fig, ax = plt.subplots(figsize=(17, 9)) # set size
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        ax.set_title("K-Means Clustering")
        #iterate through groups to layer the plot
        #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
        for name, group in groups:
            ax.plot(group.x, group.y, marker='d', linestyle='', ms=12,
                    label=cluster_names[name], color=cluster_colors[name],
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params(\
                axis= 'x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            ax.tick_params(\
                axis= 'y',         # changes apply to the y-axis
                which='both',      # both major and minor ticks are affected
                left='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelleft='off')

        ax.legend(numpoints=1, loc='upper left')  #show legend with only 1 point

        #add label in x,y position with the label as the film title
        for i in range(len(df)):
            ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'], size=7)

        fig.set_tight_layout(True)  # show plot with tight layout
        plt.savefig('./k_means.pdf', format='pdf')
        plt.show() #show the plot

    def hierachical_clustering(self):
        linkage_matrix = ward(self.__dist_matrix) #define the linkage_matrix using ward clustering pre-computed distances

        fig, ax = plt.subplots(figsize=(17, 9)) # set size
        ax.set_title("Hierachical Clustering")
        dendrogram(linkage_matrix, orientation="left", labels=titles);

        plt.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')

        fig.set_tight_layout(True) #show plot with tight layout
        plt.savefig('./hierachial.pdf', format='pdf')
        plt.show()

    def mean_shift_clustering(self):
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        pos = mds.fit_transform(self.__dist_matrix)  # shape (n_components, n_samples) 24*2

        bandwidth = estimate_bandwidth(pos, quantile=0.2, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(pos)

        labels = ms.labels_ # the length is number of clusters
        cluster_centers = ms.cluster_centers_ # 存储每个类的中心点，the length is number of clusters
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        print("number of estimated clusters : %d" % n_clusters_)
        self.show_clusters(pos,n_clusters_,cluster_centers,labels)

    def show_clusters(self,pos,n_clusters_,cluster_centers,labels,radius=2.0):
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        cluster_names = {0: 'Group 1',
                         1: 'Group 2',
                         2: 'Group 3',
                         3: 'Group 4',
                         4: 'Group 5',
                         5: 'Group 6',
                         6: 'Group 7',
                         7: 'Group 8',
                         8: 'Group 9',}
        # plt.figure(1)
        # plt.clf()
        fig, ax = plt.subplots(figsize=(17, 9))
        ax.set_title('Estimated number of clusters: %d' % n_clusters_)
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        df = pd.DataFrame(dict(x=pos[:, 0], y=pos[:, 1], title=titles))  # 3 columns: x,y,title
        for i in range(len(df)):
            ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'], size=7)
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k # 布尔索引，the length is number of clusters, 控制属于该类的点显示
            # print("my_members:",my_members)
            cluster_center = cluster_centers[k]
            ax.plot(pos[my_members, 0], pos[my_members, 1], col + '.',markersize=12)
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=18, label=cluster_names[k])

            ax.tick_params( \
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom='off',  # ticks along the bottom edge are off
                top='off',  # ticks along the top edge are off
                labelbottom='off')
            ax.tick_params( \
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                left='off',  # ticks along the bottom edge are off
                top='off',  # ticks along the top edge are off
                labelleft='off')

            ax.set_aspect('auto')

        ax.legend(numpoints=1,loc='upper left')  # show legend with only 1 point
        fig.set_tight_layout(True)  # show plot with tight layout

        plt.savefig('./mean_shift.pdf', format='pdf')
        plt.show()


if __name__ == '__main__':
    my_classification = ATTextClassification()
    my_classification.run()
    my_classification.draw_wordbarchart()
    my_classification.draw_wordcloud()
    my_classification.tf_idf()
    my_classification.pos_tag()
    my_classification.named_entity()
    my_classification.nmf()
    my_classification.lda() #use tf
    my_classification.mean_shift_clustering()
    my_classification.k_mean_clustering(5)
    my_classification.hierachical_clustering()





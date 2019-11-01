import json
import numpy as np
import jieba
import string
import timeit
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import pickle
import os
from tqdm import  tqdm
import re

def save_obj(obj, path_name):
    with open(path_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path_name):
    with open(path_name, 'rb') as f:
        return pickle.load(f)

# 自定义打印方法
def print_format(str, a):
    print(str + '\n{0}\n'.format(a))

# cut_list
def cut_list(list_input, isSpecialHandle=True):
    list_new = []
    for sentence in list_input:
        if isSpecialHandle:
            list_new.append(sentence.replace('?','').split())
        else:
            list_new.append(sentence.split())
    return list_new

#handle_one_sentence
def handle_one_sentence(sentence):
    return sentence.replace('?','').split(' ')

vocab  = {line.rstrip(): None for line in open('vocab.txt')}
# vocab  = re.findall("[a-z]+", open('vocab.txt').read())

# ==============================第一部分：对于训练数据的处理：读取文件和预处理=======================

# 文本的读取： 需要从文本中读取数据，此处需要读取的文件是dev-v2.0.json，并把读取的文件存入一个列表里（list）
def read_corpus():
    #解析json数据
    #Tips1:答案字典“answers”可能会有空的情况 此时应该取plausible_answers节点
    qlist = []
    alist = []

    filename = 'train-v2.0.json'
    with open(filename,'r') as load_f:
        load_dict = json.load(load_f)
        data_list = load_dict['data']
        # len_data = len(data_list)
        # print_format("len_data", len_data)
        for data in data_list:
            paragraphs = data["paragraphs"]
            for paragraph in paragraphs:
               qas = paragraph["qas"]
               for qa in paragraph["qas"]:
                   if "answers" in qa:
                       if len(qa["answers"]) > 0 != None and qa["answers"][0]["text"] != None:
                           qlist.append(qa["question"])
                           alist.append(qa["answers"][0]["text"])


    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, np.array(alist)

#### 1.2 理解数据（可视化分析/统计信息）

qlist, alist = read_corpus()
word_total = [word for words_list in cut_list(qlist) for word in words_list]
word_total_old = [word for word in word_total]
word_total_unique = list(set(word_total))


# 统计词频
dict_word_count = {l:0 for l in word_total_unique}
for value in word_total:
    dict_word_count[value] +=1


#统计出现1,2,3...n的单词的个数
word_count_set = sorted(list(set(dict_word_count.values())))
dict_appear_counts = {s:0 for s in word_count_set}
for item in dict_word_count.items():
    dict_appear_counts[item[1]] += 1

x_data = list(dict_appear_counts.keys())
y_data = list(dict_appear_counts.values())

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)

fig = plt.figure()  #设置画布
ax1 = fig.add_subplot(111)
k = 50
plt.plot(x_data[:k], y_data[:k])
ax1.set_xlabel(u'单词出现的次数', fontproperties=font_set)
ax1.set_ylabel(u'单词个数', fontproperties=font_set)
plt.show()

fig = plt.figure()  #设置画布
ax1 = fig.add_subplot(111)
ax1.hist(x_data, range=(0,2000),bins=30)
plt.show()

# ======================================1.3 文本预处理=================================================

# print(qlist[0:10])
stopwords = {line.rstrip().lower(): None for line in open('stopwords.txt')}
# remove_punct_map = {c: None for c in string.punctuation}
remove_punct_map = {c: None for c in '?!@#$%^&*()~`,./|'}

low_freg_words = {value[0]: None for value in dict_word_count.items() if value[1] < 3}

# 对单词进行处理
def handle_one_word(word, isUseLowFreg = True, isUseStopWord = True):

    # 过滤掉频率低的单词
    if isUseLowFreg:
        if word in low_freg_words:
            return None


    # 去除所有标点符号
    word = ''.join(c for c in word if c not in remove_punct_map)

    # 停用词过滤
    if isUseStopWord:
        if word.lower() in stopwords:
            # print(word)
            return None

    # 处理数字
    if word.isdigit():
        word = word.replace(word, "#number")

    # 单词转小写
    word = word.lower()

    return word


def text_preparation(qlist):

    """
    - 1. 停用词过滤 （去网上搜一下 "english stop words list"，会出现很多包含停用词库的网页，或者直接使用NLTK自带的）
    - 2. 转换成lower_case： 这是一个基本的操作
    - 3. 去掉一些无用的符号： 比如连续的感叹号！！！， 或者一些奇怪的单词。
    - 4. 去掉出现频率很低的词：比如出现次数少于10,20.... （想一下如何选择阈值）
    - 5. 对于数字的处理： 分词完只有有些单词可能就是数字比如44，415，把所有这些数字都看成是一个单词，这个新的单词我们可以定义为 "#number"
    """
    start = timeit.default_timer()
    qlist_new = []

    for sentence in qlist:
        sentence_new = ''
        words_list = handle_one_sentence(sentence)

        for word in words_list:

            word = handle_one_word(word)
            if word != None:
                sentence_new += word + " "

        qlist_new.append(sentence_new)


    # qlist = qlist_new
    qlist= [q for q in qlist_new if q.rstrip() != ""]


    stop = timeit.default_timer()
    print('文本预处理 Time: ', stop - start)

    return qlist

qlist = text_preparation(qlist)
cut_table = cut_list(qlist, isSpecialHandle=False)
word_total = [word for words_list in cut_table for word in words_list]
word_total_unique = list(set(word_total))


# =================================tfidf 方式  开始=========================================

vectorizer = TfidfVectorizer()  # 定一个tf-idf的vectorizer
def TfidfVector_Sklean():
    start = timeit.default_timer()
    X_tfidf = vectorizer.fit_transform(qlist)   # 结果存放在X矩阵
    stop = timeit.default_timer()
    print('TfidfVector_Sklean Time: ', stop - start)
    return X_tfidf

X_tfidf = TfidfVector_Sklean()


# =====================================tfidf 方式  结束============================================

def GetSentenceVectorCommon(sentence, embedding_dict, dim, isUseAveragePooling = False):
    total_effect_count = 0
    w_v = []
    for word in sentence:
        if word in embedding_dict:
            total_effect_count += 1
            w_v.append(embedding_dict[word])

    w_v = np.array(w_v)

    is_effect = total_effect_count > 0
    if  is_effect:
        if isUseAveragePooling:
            w_v = np.sum(w_v, axis=0) / total_effect_count
        else:
            w_v = np.max(w_v, axis=0)
    else:
        w_v = np.zeros(dim)

    return np.array(w_v)

# =====================================glove 方式  开始============================================

embeddings_index = {}
glovefile = open("glove.6B.200d.txt", "r", encoding="utf-8")
glove_store_path_name = "glove200_dict.pkl"

for line in glovefile:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float16')
    embeddings_index[word] = coefs
glovefile.close()

embedding_dim_glove = 200
def get_embedding_matrix_glove(word):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        return embedding_vector[:embedding_dim_glove]
    return np.zeros(embedding_dim_glove)

word2id = {}
emd = []
for word in word_total_unique:
    if word not in word2id:
        word2id[word] = len(word2id)
        emd.append(get_embedding_matrix_glove(word))
emd = np.asarray(emd)


X_w2v = []
for sentence in cut_table:
    w_v = GetSentenceVectorCommon(sentence, embeddings_index, embedding_dim_glove, isUseAveragePooling=True)
    X_w2v.append(w_v)

X_w2v = np.asarray(X_w2v)



# =================================glove 方式  结束=========================================

# =====================================bert 方式  开始============================================

from bert_embedding import BertEmbedding
import mxnet

# 如果有gpu的话
# ctx = mxnet.gpu()
# bertEmbedding = BertEmbedding(ctx=ctx, model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased')
# bertEmbedding = BertEmbedding(ctx=ctx, model='bert_12_768_12')
bertEmbedding = BertEmbedding(model='bert_12_768_12')

def get_bert_embedding(qlist, isUseAveragePool = True):
    """
    功能：获取bert词向量
    :param qlist: 文本list,
    :return: 每个句子的平均bert词向量
    """
    embedding_list = []

    # qlist_new = [q.split('\n') for q in qlist]

    # all_embedding = bertEmbedding(qlist)
    # for index, item in tqdm(enumerate(all_embedding)):
    #     average_ques_embedding[index] = np.sum(item[1], axis=0) / len(item[0])
    #
    # return average_ques_embedding

    for index, question in enumerate(qlist):
        result = bertEmbedding(question.split('\n'))
        item = result[0]
        if len(item[0]) > 0:
            if isUseAveragePool:
                embedding_list.append(np.sum(np.array(item[1]), axis=0) / len(item[0]))
            else:
                embedding_list.append(np.max(np.array(item[1]), axis=0))

    return np.array(embedding_list, dtype= np.float16)

xbert_vector_s_p = "xbert_vector_new.npy"

X_bert = np.zeros(shape=(len(qlist), 768), dtype= np.float16)
if not os.path.exists(xbert_vector_s_p):
    X_bert = get_bert_embedding(qlist, False)
    np.save(xbert_vector_s_p, X_bert)
else:
    X_bert = np.load(xbert_vector_s_p)

X_bert = np.array(X_bert)

# =====================================bert 方式 结束============================================


# =====================================第3大部分相似度匹配  开始===================================================

import heapq
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse

def get_least_numbers_big_data(alist, k):
    max_heap = []
    length = len(alist)
    if not alist or k <= 0 or k > length:
        return
    k-=1
    for ele in alist:
        if len(max_heap) <= k:
            heapq.heappush(max_heap, ele)
        else:
            heapq.heappushpop(max_heap, ele)

    # return list(map(lambda x:x, max_heap))
    return max_heap

def get_top_results_tfidf_noindex(query):
    # TODO 需要编写
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 对于用户的输入 query 首先做一系列的预处理(上面提到的方法)，然后再转换成tf-idf向量（利用上面的vectorizer)
    2. 计算跟每个库里的问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """

    input_seq = query
    input_vec = vectorizer.transform([input_seq])
    result = list(cosine_similarity(input_vec, X_tfidf)[0])
    top_values = sorted(get_least_numbers_big_data(result, 5), reverse=True)

    top_idxs = []
    len_result = len(result)
    dict_visited = {}
    for value in top_values:
        for index in range(len_result):
            if value == result[index] and index not in dict_visited:
                top_idxs.append(index)
                dict_visited[index] = True

    top_idxs = top_idxs[:5]

    return alist[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案

# print_format("When did beyonce get popular",get_top_results_tfidf_noindex("When did beyonce get popular"))
# print_format("In what city and state did Beyonce  grow up?",get_top_results_tfidf_noindex("In what city and state did Beyonce  grow up?"))
# print_format("What areas did Beyonce compete in when she was growing up?",get_top_results_tfidf_noindex("What areas did Beyonce compete in when she was growing up?"))

# =====================================第3大部分相似度匹配  结束===================================================


# =====================================倒排表的创建  开始===================================================
# TODO 请创建倒排表
inverted_idx =  {value:[] for value in word_total_unique}  # 定一个一个简单的倒排表，是一个map结构。 循环所有qlist一遍就可以
for index, sentence in enumerate(cut_table):
    # print(index, sentence)
    for word in sentence:
        inverted_idx[word].append(index)
# print_format("inverted_idx", inverted_idx)



# =====================================倒排表的创建  结束===================================================


#======================================相似度实现 开始======================================================
# TODO 语义相似度
def get_related_words(file):
    dict_related = {}
    for line in open(file, mode='r', encoding='utf-8'):
        item = line.split(",")
        word, si_list = item[0], [value for value in item[1].strip().split()]
        dict_related[word] = si_list
    return dict_related

related_words = get_related_words("related_words.txt")
# print_format("related_words", related_words)

def get_handled_input_seq(query):
    # return [''.join(c for c in word if c not in remove_punct_map) for word in query.split() if word.lower() not in stopwords]
    result = []
    for word in query.split():
        word = handle_one_word(word)
        if word != None:
            result.append(word)
    return result

# 检查输入的问题并返回处理过的问题tf-idf用
def check_query(query):
    input_seq = get_handled_input_seq(query)
    result = ""
    for word in input_seq:
        result += word + " "
    return result.strip()


# 利用倒排表和同义词获取相关的预料库中问题的序号
def get_related_sentences(query):
    input_seq = get_handled_input_seq(query)
    si_list = []
    for word in input_seq:
        if word in related_words:
            for value in related_words[word]:
                si_list.append(value)

    total_list = input_seq
    for word in si_list:
        total_list.append(word)

    sentence_list = []
    for word in total_list:
        if word in inverted_idx:
            sentence_list.extend(inverted_idx[word])
    return list(set(sentence_list))

def getTopIndexByResult(result):
    top_idxs = []
    top_values = sorted(get_least_numbers_big_data(result, 5), reverse=True)
    len_result = len(result)
    dict_visited = {}
    for value in top_values:
        for index in range(len_result):
            if value == result[index] and index not in dict_visited:
                top_idxs.append(index)
                dict_visited[index] = True
    return top_idxs

def get_top_results_tfidf(query):
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate （需要使用related_words).
    2. 对于候选文档，计算跟输入问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """

    query = check_query(query)
    if query == "":
        print_format("please input a effect question","")
        return None

    sentence_list = get_related_sentences(query)

    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表

    input_seq = query
    input_vec = vectorizer.transform([input_seq])

    is_use_s_l = len(sentence_list) > 0

    if is_use_s_l == True:
        X_tfidf_si = []
        for id in sentence_list:
            X_tfidf_si.append(X_tfidf[id].toarray()[0])
        X_tfidf_si = np.array(X_tfidf_si)

        result = list(cosine_similarity(input_vec, csr_matrix(X_tfidf_si))[0])
    else:
        result = list(cosine_similarity(input_vec, X_tfidf)[0])

    top_idxs = getTopIndexByResult(result)

    if is_use_s_l == True:
        top_idxs = [sentence_list[idx] for idx in top_idxs[:5]]
    else:
        top_idxs = top_idxs[:5]

    return alist[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


# print("\n")
# print_format("When did beyonce get popular",get_top_results_tfidf("When did beyonce get popular"))
# print_format("What areas did Beyonce compete in when she was growing up?",get_top_results_tfidf("What areas did Beyonce compete in when she was growing up?"))
# print_format("In what city and state did Beyonce  grow up?",get_top_results_tfidf("In what city and state did Beyonce grow up?"))
# print("\n")

def get_top_results_w2v(query):
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate （需要使用related_words).
    2. 对于候选文档，计算跟输入问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """

    query = check_query(query)
    if query == "":
        print_format("please input a effect question","")
        return None

    sentence_list = get_related_sentences(query)

    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表

    input_seq = get_handled_input_seq(query)
    seq_new = " ".join(word for word in input_seq)

    g_wv = GetSentenceVectorCommon(input_seq, embeddings_index, embedding_dim_glove, isUseAveragePooling=True)
    # g_wv = compute_each_sentence_embedding(seq_new, glove_model)

    is_use_s_l = len(sentence_list) > 0

    if is_use_s_l == True:
        X_glove_si = []
        for id in sentence_list:
            X_glove_si.append(X_w2v[id])
        X_glove_si = np.array(X_glove_si)
        result = list(cosine_similarity(csr_matrix(g_wv), csr_matrix(X_glove_si))[0])
    else:
        result = list(cosine_similarity(csr_matrix(g_wv), csr_matrix(X_w2v))[0])

    top_idxs = getTopIndexByResult(result)

    if is_use_s_l == True:
        top_idxs = [sentence_list[idx] for idx in top_idxs[:5]]
    else:
        top_idxs = top_idxs[:5]

    return alist[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案

# print_format("When did beyonce get popular",get_top_results_w2v("When did beyonce get popular"))
# print_format("What areas did Beyonce compete in when she was growing up?",get_top_results_w2v("What areas did Beyonce compete in when she was growing up?"))
# print_format("In what city and state did Beyonce  grow up?",get_top_results_w2v("In what city and state did Beyonce grow up?"))


# bert 方式获取相似度最高的答案
def get_top_results_bert(query):
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate （需要使用related_words).
    2. 对于候选文档，计算跟输入问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """

    query = check_query(query)
    if query == "":
        print_format("please input a effect question","")
        return None

    sentence_list = get_related_sentences(query)

    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表

    input_seq = get_handled_input_seq(query)

    # words_unique = list(set(input_seq))
    # bert_emd = {}
    # for word in words_unique:
    #     sentences = word.split('\n')
    #     result = bert_embed(sentences)
    #     first_sentence = result[0]
    #     if len(first_sentence[1]) > 0:
    #         bert_emd[word] = first_sentence[1][0]
    #
    # b_wv = GetSentenceVectorCommon(input_seq, bert_emd, embedding_dim_bert, isUseAveragePooling=False)

    seq_new = " ".join(word for word in input_seq)
    b_wv = get_bert_embedding([seq_new], False)

    is_use_s_l = len(sentence_list) > 0

    if is_use_s_l == True:
        X_bert_si = []
        for id in sentence_list:
            X_bert_si.append(X_bert[id])
        X_bert_si = np.array(X_bert_si)
        result = list(cosine_similarity(csr_matrix(b_wv), csr_matrix(X_bert_si))[0])
    else:
        result = list(cosine_similarity(csr_matrix(b_wv), csr_matrix(X_bert))[0])

    top_idxs = getTopIndexByResult(result)

    if is_use_s_l == True:
        top_idxs = [sentence_list[idx] for idx in top_idxs[:5]]
    else:
        top_idxs = top_idxs[:5]

    return alist[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


# print_format("When did beyonce get popular",get_top_results_bert("When did beyonce get popular"))
# print_format("What areas did Beyonce compete in when she was growing up?",get_top_results_bert("What areas did Beyonce compete in when she was growing up?"))
# print_format("In what city and state did Beyonce  grow up?",get_top_results_bert("In what city and state did Beyonce grow up?"))

# question_list = ["   ",
#                  "!!$$$%%",
#                  "When did beyonce get popular",
#                  "What areas did Beyonce compete in when she was growing up?",
#                  "For which song, did Destiny's Child take home the grammy award for best R&B performance?",
#                  "Beyonce's group changed their name to Destiny's Child in what year?",
#                  "Destiny's Child song, Killing Time, was included in which film's soundtrack?",
#                  "What"]

question_list = ["how much more that the but did the film ? ",
                 "what birthday did abeyance 's album celebrate?"]

def test_output(question_list, iscorrect=False):
    for question in question_list:
        if question.strip() == "":
            print("Your question is empty")
            continue

        if iscorrect:
            question = spell_corrector(question)

        # input_seq = get_handled_input_seq(question)
        # if check_query(question) == "":
        #     print_format("please input a effective question", question)
        #     continue

        print_format("===============" + question + "===================", "")
        print_format("tif:", get_top_results_tfidf(question))
        print_format("glove:", get_top_results_w2v(question))
        print_format("bert:",get_top_results_bert(question))

# test_output(question_list, iscorrect=False)

#======================================相似度实现 结束======================================================

#======================================拼写纠错 开始=========================================================




"""
构建Channel Probs
基于spell_errors.txt文件构建channel probability
其中channel[c][s]channel[c][s]表示正确的单词cc被写错成ss的概率
"""
channel = {}

spell_error_dict = {}

for line in open('spell-errors.txt'):
    item = line.split(":")
    word = item[0].strip()
    spell_error_list = [word.strip( )for word in item[1].strip().split(",")]
    spell_error_dict[word] = spell_error_list
# print_format("spell_error_dict", spell_error_dict)

for key in spell_error_dict:
    if key not in channel:
        channel[key] = {}
        for value in spell_error_dict[key]:
            channel[key][value] = 1 / len(spell_error_dict[key])
# print_format("channel", channel)

alphabet = "abcdefghijklmnopqrstuvwxyz"

import collections
def get_word_N():
    model = collections.defaultdict(lambda: 0)
    for value in word_total_old:
        model[value] += 1
    return model
words_N = get_word_N()
# print_format("len(words_unique_dict)", len(words_unique_dict))

def known(words):
    # return list(set(w for w in words if w in words_N))
    return list(set(w for w in words if w in vocab))

def edits1(word):
    n = len(word)
    #删除
    s1 = [word[0:i] + word[i+1:] for i in range(n)]
    #调换相连的两个字母
    s2 = [word[0:i] + word[i+1] + word[i] + word[i+2:] for i in range(n-1)]
    #replace
    s3 = [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet]
    #插入
    s4 = [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet]
    edit1_words = set(s1 + s2 + s3 + s4)

    if word in edit1_words:
        edit1_words.remove(word)

    edit1_words = known(edit1_words)
    return edit1_words

def edits2(word, edit1_words):
    edit2_words = set(e2 for e1 in edit1_words for e2 in edits1(e1))
    if word in edit2_words:
        edit2_words.remove(word)
    edit2_words = known(edit2_words)
    return edit2_words

"""根据错别字生成所有候选集合"""
def generate_candidates(word):
    edit1_words = edits1(word)     # 编辑距离为1的候选项
    edit2_words = edits2(word, edit1_words)   # 编辑距离为2的候选项
    candidates = edit1_words
    for word in edit2_words:
        candidates.append(word)
    return candidates

# print_format("generate_candidates('looking')", generate_candidates('looking'))
# print("cooking" in words_unique_dict)


from nltk.corpus import reuters

# 读取语料库的数据
categories = reuters.categories()
corpus = reuters.sents(categories=categories)
# 循环所有的语料库并构建bigram probability. bigram[word1][word2]: 在word1出现的情况下下一个是word2的概率。

# print_format("categories['from']", categories['from'])
# print_format("len(corpus)", len(corpus))

word_count_dict = collections.defaultdict(lambda: 0)
word_bigram_dict = {}
# test_index = 0
for corpu in corpus:
    # if "eat" in corpu:
    #     print(corpu)
    for index, word in enumerate(corpu):
        word_count_dict[word] += 1
        if index >= 1:
            prev_word = corpu[index - 1]
            if prev_word not in word_bigram_dict:
                word_bigram_dict[prev_word] = {}
            if word not in word_bigram_dict[prev_word]:
                word_bigram_dict[prev_word][word] = 0
            word_bigram_dict[prev_word][word] += 1
    # test_index + 1

# print_format("word_count_dict", word_count_dict)
# print_format("word_bigram_dict", word_bigram_dict)

# print(word_bigram_dict["eat"]["less"], word_count_dict["eat"])

V_words = len(word_count_dict.keys())

# print_format("V_words", V_words)

for key_bigram in word_bigram_dict.keys():
    for key_word in word_bigram_dict[key_bigram].keys():
        word_bigram_dict[key_bigram][key_word] = (word_bigram_dict[key_bigram][key_word] + 1) / (word_count_dict[key_bigram] + V_words)

# print_format("len(word_bigram_dict)", len(word_bigram_dict))
# print_format("word_bigram_dict['eat']['less']", word_bigram_dict['eat']["less"])

def getCorrectestWord(input_word, token_list, cur_index):
    candidates = generate_candidates(input_word.lower())

    if len(candidates) == 0:
        return input_word

    candidates_spell_error = []
    for candidate in candidates:
        if candidate in channel and input_word in channel[candidate]:
            candidates_spell_error.append(candidate)

    if len(candidates_spell_error) == 0:
        return max(candidates, key=lambda w: words_N[w])

    candidates = candidates_spell_error

    if(len(token_list) == 1):
        return max(candidates, key = lambda w: channel[w][input_word])

    bein_pos_state = cur_index == 0
    middle_pos_state = cur_index > 0 and  cur_index < len(token_list) - 1
    end_pos_state = cur_index == len(token_list) - 1

    prev_word = token_list[cur_index - 1] if cur_index > 0 else None
    next_word = token_list[cur_index + 1] if cur_index < len(token_list) - 1 else None

    candidates_bigram = []

    candidates_bigram_value_dic = {candidate:0 for candidate in candidates}

    for candidate in candidates:
        is_bigram_right = False
        is_bigram_left = False

        if bein_pos_state or middle_pos_state:
            if bein_pos_state:
                is_bigram_left = True
            if candidate in word_bigram_dict and next_word != None and next_word in word_bigram_dict[candidate]:
                is_bigram_right = True

        if end_pos_state or middle_pos_state:
            if end_pos_state:
                is_bigram_right = True
            if prev_word in word_bigram_dict and prev_word != None and candidate in word_bigram_dict[prev_word]:
                is_bigram_left = True
        is_bigram = is_bigram_left and is_bigram_right
        if is_bigram:
            candidates_bigram.append(candidate)
            bigram_left_prob = 1 if bein_pos_state else word_bigram_dict[prev_word][candidate]
            bigram_right_prob = 1 if end_pos_state else word_bigram_dict[candidate][next_word]
            candidates_bigram_value_dic[candidate] = bigram_left_prob * bigram_right_prob

    if len(candidates_bigram) == 0:
        return max(candidates, key=lambda w: channel[w][input_word])

    candidates = candidates_bigram

    return max(candidates, key=lambda w: channel[w][input_word] * candidates_bigram_value_dic[candidate])



def spell_corrector(line):

    # 1. 首先做分词，然后把``line``表示成``tokens``
    # 2. 循环每一token, 然后判断是否存在词库里。如果不存在就意味着是拼写错误的，需要修正。
    #    修正的过程就使用上述提到的``noisy channel model``, 然后从而找出最好的修正之后的结果。

    tokens = [word.strip() for word in line.split()]

    newline = ""
    for index, token in enumerate(tokens):
        token = handle_one_word(token, isUseLowFreg=False, isUseStopWord=False)
        if token == None or token.strip() == "":
            continue
        # if token.lower() not in words_N: #默认单词拼错了
        if token.lower() not in vocab:  # 默认单词拼错了
            token = getCorrectestWord(token, tokens, index)
        newline += token + " "
    return newline


# print_format("spell_corrector('get populer')", spell_corrector('get populer'))
# print_format("spell_corrector('populer')", spell_corrector('populer'))
# print_format("spell_corrector('populer kooo')", spell_corrector('populer kooo'))
# print_format("spell_corrector('populer get')", spell_corrector('populer get'))
# print(spell_corrector("Beyonce's group chaged their name to Destiny's Child in whhat year?"))

# print(generate_candidates("poular"))

question_list_1 = ["In whhat citty and state did Beyonce grow up?",
                 "When did beyonce get populer",
                 "For which song, did Destiny's Child tak home the grammy awar for best R&B performance?",
                 "Beyonce's group chaged their name to Destiny's Child in whhat year?",
                 "how much more that the but did the film gross"
                ]

# question_list_1 = ["how much more that the but did the film gross ?",
#                  "what birthday did abeyance 's album celebrate ?"]

# question_list_1 = text_preparation(question_list_1)

test_output(question_list_1, iscorrect=True)

#======================================拼写纠错 结束=========================================================
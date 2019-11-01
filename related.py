import json
import numpy as np
import jieba
import string
import timeit
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import os

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


qlist, alist = read_corpus()
word_total = [word for words_list in cut_list(qlist) for word in words_list]
word_total_unique = list(set(word_total))

# print_format("len(word_total)", len(word_total))


# 统计词频
dict_word_count = {l:0 for l in word_total_unique}
for value in word_total:
    dict_word_count[value] +=1

def text_preparation(qlist):

    """
    - 1. 停用词过滤 （去网上搜一下 "english stop words list"，会出现很多包含停用词库的网页，或者直接使用NLTK自带的）
    - 2. 转换成lower_case： 这是一个基本的操作
    - 3. 去掉一些无用的符号： 比如连续的感叹号！！！， 或者一些奇怪的单词。
    - 4. 去掉出现频率很低的词：比如出现次数少于10,20.... （想一下如何选择阈值）
    - 5. 对于数字的处理： 分词完只有有些单词可能就是数字比如44，415，把所有这些数字都看成是一个单词，这个新的单词我们可以定义为 "#number"
    - 6. lemmazation： 在这里不要使用stemming， 因为stemming的结果有可能不是valid word。
    """

    stopwords = {line.rstrip().lower():None for line in open('stopwords.txt')}

    low_freg_words = {value[0]:None for value in dict_word_count.items() if value[1] < 3}

    start = timeit.default_timer()
    qlist_new = []

    remove_punct_map = {c:None for c in string.punctuation}

    for sentence in qlist:
        sentence_new = ''
        words_list = handle_one_sentence(sentence)
        for word in words_list:

            # 过滤掉频率低的单词
            if word in low_freg_words:
                continue

            # 去除所有标点符号
            word = ''.join(c for c in word if c not in remove_punct_map)


            #停用词过滤
            if word.lower() in stopwords:
                # print(word)
                continue

            # 处理数字
            if word.isdigit():
                word = word.replace(word, "#number")

            #单词转小写
            word = word.lower()

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

print_format("len(word_total_unique)", len(word_total_unique))

# =====================================glove 方式  开始============================================

embeddings_index = {}
glovefile = open("glove.6B.200d.txt", "r", encoding="utf-8")

for line in glovefile:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float16')
    embeddings_index[word] = coefs
glovefile.close()

embedding_dim = 200
def get_embedding_matrix_glove(word):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        return embedding_vector[:embedding_dim]
    return np.zeros(embedding_dim)

word2id, id2word = {}, {}
emd = []
for word in word_total_unique:
    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
        emd.append(get_embedding_matrix_glove(word))
emd = np.asarray(emd)

dict_related = {word:[] for word in word_total_unique}
emd_csr_matrix = scipy.sparse.csr_matrix(emd)

test_count = 0

for key in dict_related.keys():
    # if test_count >=100:
    #     break
    # test_count += 1

    word_index = word2id[key]

    # v_k = emd[index]
    # result = list(cosine_similarity(scipy.sparse.csr_matrix(v_k), emd_csr_matrix)[0])

    result = list(cosine_similarity(emd_csr_matrix[word_index], emd_csr_matrix)[0])

    top_values = sorted(get_least_numbers_big_data(result, 10), reverse=True)

    top_idxs = []
    len_result = len(result)
    dict_visited = {}
    for value in top_values:
        for i in range(len_result):
            if value == result[i] and i not in dict_visited and word_index != i:
                top_idxs.append(i)
                dict_visited[i] = True

    top_idxs = top_idxs[:10]

    word_total_unique = np.array(word_total_unique)
    dict_related[key] = list(word_total_unique[top_idxs])


# print("dict_related", dict_related)

file_store_path = 'related_words.txt'
if os.path.exists(file_store_path):
    os.remove(file_store_path)

# file = open(file_store_path, 'w')
# file.writelines('你好，\n  hello。')
# file.close()

with open(file_store_path, mode='w', encoding='utf-8') as file:
    # file.writelines('你好，\n  hello。')
    for item in dict_related.items():
        r_l = " ".join(word for word in item[1])
        output = '{0},{1}'.format(item[0], r_l)
        file.write(output + "\n")
    file.close()





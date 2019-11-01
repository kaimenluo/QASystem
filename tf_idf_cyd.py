import numpy as np
import time

class TfidfVector_C(object):
    def __init__(self, input_data, cut_table):
        # cut_table = cut_list(input_data, isSpecialHandle=False)
        word_total = [word for words_list in cut_table for word in words_list]
        word_total_unique = list(set(word_total))

        self.dict_idf = {word: 0 for word in word_total_unique}
        self.dict_column_index = {item[0]: index for index, item in enumerate(self.dict_idf.items())}

    def fit_transform(self, input_data):
        # start = timeit.default_timer()
        len_qlist = len(input_data)
        list_sentence = []

        for sentence in input_data:
            word_list = sentence.split()
            len_sentence = len(word_list)
            words_list_set = set(word_list)
            words_list_dict = {}

            for word in word_list:
                if word not in words_list_dict:
                    if word not in self.dict_column_index:
                        words_list_dict[word] = [0, 0]
                    else:
                        words_list_dict[word] = [0, self.dict_column_index[word]]
                words_list_dict[word][0] += 1

            for key in words_list_dict.keys():
                words_list_dict[key][0] /= len_sentence

                if key in self.dict_idf:
                    self.dict_idf[key] += 1

            list_sentence.append(words_list_dict)

        for key in self.dict_idf.keys():
            if self.dict_idf[key] > 0:
                self.dict_idf[key] = -np.log(len_qlist / self.dict_idf[key])

        X_tfidf = np.zeros((np.int(len_qlist), np.int(len(self.dict_idf.keys()))), dtype="float16")
        for index, value in enumerate(list_sentence):
            for item in value.items():
                word = item[0]

                if word not in self.dict_column_index:
                    continue

                tf_of_word = item[1][0]
                column_index = item[1][1]
                X_tfidf[index, column_index] = np.float16(tf_of_word * self.dict_idf[word])

        # stop = timeit.default_timer()
        # print('TfidfVector_Implenet Time: ', stop - start)
        return X_tfidf

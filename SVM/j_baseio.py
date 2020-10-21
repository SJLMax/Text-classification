import codecs
import os
import json
import pickle
import random
import time
from functools import wraps

CHINESE_PUNCTUATION = list('，。、；：‘’“”！？《》「」【】<>（）')
ENGLISH_PUNCTUATION = list(',.;:\'"!?<>()')


# 定义装饰器
def fn_timer(function):
    @wraps(function)
    def function_timer(*args,**kwargs):
        t0 = time.time()
        result = function(*args,**kwargs)
        t1 = time.time()
        print('[finished {func_name} in {time:.2f}s]'.format(func_name = function.__name__,time = t1 - t0))
        return result
    return function_timer


def read_seq_res(path, labels):
    '''
    读序列标注三列数据的方法
    :param path:
    :param labels:
    :return:
    '''
    with codecs.open(path, 'r', 'utf-8') as rd:
        seqs_str = rd.read().strip()
    seqs_list = seqs_str.split('\n\n')
    text, raw_label, predict_label = [], [], []
    for seq in seqs_list:
        seq_split = seq.split('\n')
        text_tmp = ''
        raw_index_dict, pre_index_dict = {}, {}
        for label in labels:
            raw_index_dict.setdefault(label, [])
            pre_index_dict.setdefault(label, [])
        for idx, line in enumerate(seq_split):
            tmp = line.split('\t')
            text_tmp += tmp[0]
            if tmp[1] in labels:
                raw_index_dict[tmp[1]].append(idx)
            if tmp[2] in labels:
                pre_index_dict[tmp[2]].append(idx)
        text.append(text_tmp)
        raw_label.append(raw_index_dict)
        predict_label.append(pre_index_dict)
    return text, raw_label, predict_label


def subject_object_labeling(spo_list, text):
    # TODO
    '''
    百度那种有spo字典的数据，给标成。草，看不懂，得找找哪里用的
    :param spo_list:
    :param text:
    :return: labeling_list
    '''
    def _spo_list_to_spo_predicate_dict(spo_list):
        spo_predicate_dict = dict()
        for spo_item in spo_list:
            predicate = spo_item["predicate"]
            subject = spo_item["subject"]
            object = spo_item["object"]
            spo_predicate_dict.setdefault(predicate, []).append((subject, object))
        return spo_predicate_dict

    def _index_q_list_in_k_list(q_list, k_list):
        """Known q_list in k_list, find index(first time) of q_list in k_list"""
        q_list_length = len(q_list)
        k_list_length = len(k_list)
        for idx in range(k_list_length - q_list_length + 1):
            t = [q == k for q, k in zip(q_list, k_list[idx: idx + q_list_length])]
            # print(idx, t)
            if all(t):
                # print(idx)
                idx_start = idx
                return idx_start

    def _labeling_type(spo, spo_type):
        idx_start = _index_q_list_in_k_list(q_list=spo, k_list=text)
        labeling_list[idx_start] = 'B-' + spo_type
        if len(spo) == 2:
            labeling_list[idx_start + 1] = 'I-' + spo_type
        elif len(spo) >= 3:
            labeling_list[idx_start + 1: idx_start + len(spo)] = ['I-' + spo_type] * (len(spo) - 1)
        else:
            pass

    spo_predicate_dict = _spo_list_to_spo_predicate_dict(spo_list)
    labeling_list = ['O'] * len(text)
    # count = 0
    for predicate, spo_list_form in spo_predicate_dict.items():
        if predicate in text:
            for (spo_subject, spo_object) in spo_list_form:
                # if predicate not in spo_subject and predicate not in spo_object:
                _labeling_type(spo_subject, 'SUB')
                _labeling_type(spo_object, 'OBJ')
                _labeling_type(predicate, 'PRE')
                # count += 1
                # print(count)
                # if count == 2:
                #     print()
            if labeling_list != ['O'] * len(text):
                return labeling_list
    return None


def label(text, labels):
    '''
    返回两列的标记数据序列
    :param text:
    :param labels:
    :return:
    '''
    train_sequence = '\n'.join(['\t'.join(i) if i[0] != ' ' else '[null]\tO' for i in zip(list(text), labels)])
    return train_sequence


# line by line
def save_to_json(corpus, path):
    with open(path, 'w', encoding='utf-8') as wt:
        for i in corpus:
            wt.write(json.dumps(i, ensure_ascii=False))
            wt.write('\n')


# line by line
def load_from_json(path):
    with open(path, 'r', encoding='utf-8') as rd:
        corpus = []
        while True:
            line = rd.readline()
            if line:
                corpus.append(json.loads(line))
            else:
                break
    return corpus


def kfold(corpus, path, k=9, is_shuffle=True):
    '''
    k是10份中训练集占了几份
    '''
    j_mkdir(path)
    if is_shuffle:
        random.shuffle(corpus)
    split_position = int(len(corpus) / 10)
    train_set, dev_set = corpus[:k * split_position], corpus[k * split_position:]
    writetxt_a_list(train_set, os.path.join(path, 'train.txt'))
    writetxt_a_list(dev_set, os.path.join(path, 'test.txt'))
    writetxt_a_list(dev_set, os.path.join(path, 'dev.txt'))


###########################################################################    os
def j_mkdir(name):
    if not os.path.exists(name):
        os.mkdir(name)


def get_filename(path):
    '''
    返回路径最后的文件名
    :param path:
    :return:
    '''
    # path = r'哲学-已标记208本-抽取后人工校正版20200316/2哲学-已标记/中国传统价值观诠释学（刘翔）.txt'
    filename = os.path.split(path)[-1]
    return filename

# TODO 还没写
def walk():
    paths = os.walk(r'F:\python_project\tmp_data\njubook_data\标注的图书_标注数据')
    for root, dir, files in paths:
        for name in files:
            if name == 'predict.tf_record':
                os.remove(os.path.join(root, name))

def j_listdir(dir_name, including_dir=True):
    # ATTENTION yield是
    filenames = os.listdir(dir_name)
    for filename in filenames:
        if including_dir:
            yield os.path.join(dir_name, filename)
        else:
            yield filename


############################################################################   txt

# 读txt文件 一次全读完 返回list 去换行
def readtxt_list_all_strip(path, encoding='utf-8'):
    lines = []
    with codecs.open(path, 'r', encoding) as r:
        for line in r.readlines():
            line = line.strip('\n').strip("\r")
            lines.append(line)
        return lines


# 读txt 一次读一行 最后返回list
def readtxt_list_each(path):
    lines = []
    with codecs.open(path, 'r', 'utf-8') as r:
        line = r.readline()
        while line:
            lines.append(line)
            line = r.readline()
    return lines


# 读txt 一次读一行 最后返回list 去换行
def readtxt_list_each_strip(path):
    lines = []
    with codecs.open(path, 'r', 'utf-8') as r:
        line = r.readline()
        while line:
            lines.append(line.strip("\n").strip("\r"))
            line = r.readline()
    return lines


# 读txt文件 一次全读完 返回list
def readtxt_list_all(path):
    with codecs.open(path, 'r', 'utf-8') as r:
        lines = r.readlines()
        return lines



# 读byte文件 读成一条string
def readtxt_byte(path, encoding="utf-8"):
    with codecs.open(path, 'rb') as r:
        lines = r.read()
        lines = lines.decode(encoding)
        return lines.replace('\r', '')


# 读txt文件 读成一条string
def readtxt_string(path, encoding="utf-8"):
    with codecs.open(path, 'r', encoding) as r:
        lines = r.read()
        return lines.replace('\r', '')


# 写txt文件覆盖
def writetxt_w(txt, path):
    with codecs.open(path, 'w', 'utf-8') as w:
        w.writelines(txt)


# 写txt文件追加
def writetxt_a(txt, path):
    with codecs.open(path, 'a', 'utf-8') as w:
        w.writelines(txt)


def writetxt(txt, path, encoding="utf-8"):
    with codecs.open(path, 'w', encoding) as w:
        w.write(txt)


def writetxt_wb(txt, path):
    with codecs.open(path, 'wb') as w:
        w.write(txt)

# 写list 覆盖
def writetxt_w_list(list, path, num_lf=2):
    with codecs.open(path, 'w', "utf-8") as w:
        for i in list:
            w.write(i)
            w.write("\n" * num_lf)

# 写list 追加
def writetxt_a_list(list, path, num_lf=2):
    with codecs.open(path, 'a', "utf-8") as w:
        for i in list:
            w.write(i)
            w.write("\n" * num_lf)

# 写二维list 追加
def writetxt_a_2list(list, path):
    with codecs.open(path, 'a', "utf-8") as w:
        for i in list:
            writetxt_a_list(i, path)


######################################################################################
# 统计词频
def calc_word_count(list_word, mode, path='tempcount.txt', sort_id=1, is_reverse=True):
    word_count = {}
    for key in list_word:
        if key not in word_count:
            word_count[key] = 1
        else:
            word_count[key] += 1
    word_dict_sort = sorted(word_count.items(), key=lambda x: x[sort_id], reverse=is_reverse)
    if mode == 'w':
        for key in word_dict_sort:
            writetxt_a(str(key[0]) + '\t' + str(key[1]) + '\n', path)
    elif mode == 'p':
        for key in word_dict_sort:
            print(str(key[0]) + '\t' + str(key[1]))
    elif mode == 'u':
        return word_dict_sort


# 合并文件
def imgrate_files(path):
    filenames = os.listdir(path)
    return None


def SaveToJson(content, path):
    with codecs.open(path, "w", "utf-8") as w:
        json.dump(content, w)


def LoadFromJson(path):
    with codecs.open(path, "r", "utf-8") as r:
        content = json.load(r)
        return content


# 读txt文件 读成一条string if gb2312
def readtxt_string_all_encoding(path):
    try:
        with codecs.open(path, 'rb', "utf-8-sig") as r:
            lines = r.read()
            return lines
    except:
        try:
            with codecs.open(path, 'rb', "utf-8") as r:
                lines = r.reacd()
                return lines
        except:
            try:
                with codecs.open(path, 'rb', "big5") as r:
                    lines = r.read()
                    return lines
            except:
                print(path)
                with codecs.open(path, 'rb', "gb2312", errors='ignore') as r:
                    lines = r.read()
                    return lines


def readtxt_list_all_encoding(path):
    try:
        with codecs.open(path, 'rb', "utf-8-sig") as r:
            lines = r.readlines()
            return lines
    except:
        try:
            with codecs.open(path, 'rb', "utf-8") as r:
                lines = r.readlines()
                return lines
        except:
            try:
                with codecs.open(path, 'rb', "big5") as r:
                    lines = r.readlines()
                    return lines
            except:
                with codecs.open(path, 'rb', "gb2312", errors='ignore') as r:
                    lines = r.readlines()
                    return lines


def pickle_save(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def pickle_load(path):
    with open(path, 'rb') as f:

        data = pickle.load(f)
    return data

# 读取crf序列格式的数据
def read_seq_data(path):
    content = readtxt_list_all_strip(path)
    lines = [i.split('\t') if i else '' for i in content]
    print(lines)
    sequences, labels, sequence, label = [], [], [], []
    for idx, line in enumerate(lines):
        if line == '':
            if sequence:
                sequences.append(sequence)
                labels.append(label)
                sequence, label = [], []
        else:
            sequence.append(line[0])
            label.append(line[1])
        if idx == len(lines) - 1 and sequence:
            sequences.append(sequence)
            labels.append(label)
    return sequences, labels


def split_5_percent(lines):
    import random
    random.seed(8)
    # lines = list(range(1, 109))
    idx_lines = [(idx, i) for idx, i in enumerate(lines)]
    div = int(len(lines) / 100)
    sample_num = div * 5
    sample = random.sample(idx_lines, sample_num)
    sorted_sample = sorted(sample, key=lambda x:x[0])
    remove_idx = [i[0] for i in sorted_sample]
    less_has_raw_line_info = [str(i[0] + 1) + '\t' + str(i[1]) for i in sorted_sample]
    most = [i for idx,i in enumerate(lines) if not idx in remove_idx]
    print(less_has_raw_line_info)
    print(most)
    return most, less_has_raw_line_info
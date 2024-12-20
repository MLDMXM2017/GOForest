'''
Author: holyball holyball1226@gmail.com
Date: 2022-12-09 22:37:55
LastEditors: holyball holyball1226@gmail.com
LastEditTime: 2024-04-20 06:07:07
FilePath: /project/go_forest/cdutil.py
Description: 
'''
import logging

from readline import append_history_file
import time
import numpy as np
import pandas as pd
import csv
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# sys.setdefaultencoding("utf-8")

def european(mat1, mat2):
    if isinstance(mat1, np.ndarray) & isinstance(mat2, np.ndarray):
        return np.reshape(np.sum(mat1**2, axis=1), (mat1.shape[0], 1)) + np.sum(mat2**2, axis=1)-2*mat1.dot(mat2.T)
    else:
        print("mat1, mat2应该都是ndarray类型")
        return 

def cosine_distances(mat1, mat2) -> float:
    return 1-cosine_similarity(mat1, mat2)


def init_logger(logger_name: str, file_path: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    f_handler = logging.FileHandler(file_path)
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(filename)s - %(message)s"))
    logger.addHandler(f_handler)


def forest_predict(forest, x: np.ndarray, n_atom=10):
    """
    重新处理森林的预测方式：森林的输出是所有该森林中所有决策树的输出
    n_atom: 输出的增强向量的维度
    """

    if x.shape[1] != forest.n_features_in_:
        print("the shape of x not equal the n_features_in_ of the passed forest")
        return
    _estimators = forest.estimators_.copy()
    n_estimators = len(_estimators)
    step = int(n_estimators / n_atom)

    if forest is RandomForestRegressor:
        y_pred = np.array([[]] * x.shape[0])
        for i, tree in enumerate(_estimators):
            y_pred = np.hstack((y_pred, np.reshape(tree.predict(x), (-1, 1))))

        if step != 1:
            for i in range(n_atom):
                y_pred[:, i] = np.mean(y_pred[:, step * i:step * (i + 1)], axis=1)

    if forest is RandomForestClassifier:
        y_pred = np.zeros((x.shape[0], n_atom))
        for i, tree in enumerate(_estimators):
            if i % step == 0:
                temp = 0
            temp += tree.predict_proba(x)
            if temp:
                k = int(i / step - 1)
                y_pred[:, k] = np.argmax(temp, axis=1)
    return y_pred[:, 0:n_atom]


def get_idx_bins(a, bins):
    bin_idx = []
    bin_data = []
    for i in range(len(bins)):
        idx = np.intersect1d(np.argwhere(a >= bins[i]), np.argwhere(a < bins[i + 1]))  # 3类
        if i == len(bins) - 2:
            idx = np.intersect1d(np.argwhere(a >= bins[i]), np.argwhere(a <= bins[i + 1]))
            bin_idx.append(idx)
            return bin_idx
        bin_idx.append(idx)
        bin_data.append(a[idx])
    return bin_idx


def get_pred_info(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return pd.Series([mae, mape, r2], index=['mae', 'mape', 'r2'])


def write_dict_to_csv(saveDict, fileName, mode):
    with open(fileName, mode) as csv_file:
        writer = csv.writer(csv_file)
        for key, value in saveDict.items():
            # writer.writerow( [key.encode("utf-8"), [val.encode("utf-8") for val in value] ] )
            writer.writerow([key, [val for val in value]])


# 像csv文件中追加写入content
def write_log_to_csv(filepath, content):
    with open(filepath, "a") as csvfile:
        content = [content]
        writer = csv.writer(csvfile)
        writer.writerow(content)


def merge_dict(dic_a, dic_b):
    result_dic = {}
    for k, v in dic_a.items():
        for m, n in dic_b.items():
            if k == m:
                result_dic[k] = [a for a in np.array([v, n]).flat]
                # result_dic[k].append(dic_a[k])
                # result_dic[k].append(dic_b[k])
                dic_a[k] = result_dic[k]
                dic_b[k] = result_dic[k]
            else:
                result_dic[k] = dic_a[k]
                result_dic[m] = dic_b[m]
    # for key, value in result_dic.items():
    #     result_dic[key] = np.array(value).ravel()
    return result_dic


def get_currunt_time():
    return time.asctime(time.localtime(time.time()))


def getstrindex(stri, obj):
    i = 0
    index = []
    while True:
        i = stri.find(str(obj), i)

        if i == -1:
            break
        index.append(i)
        i += 1
    return index


def findgo(stri, index):
    gene_togo = [stri[i:i + 10] for i in index]
    return gene_togo


def load_train_data(train_file):
    '''
    加载训练数据, feature, label都是DataFrame或Series, feature的列为: 药物1id. 药物2id, 药物所在的细胞系
    '''
    train_data = pd.read_csv(train_file)
    feature = train_data.iloc[:, 0:-1]
    label = train_data.iloc[:, -1]

    return feature, label


def prepare_train_data(train_file):
    # '''
    # 准备训练数据
    # '''
    feature, label = load_train_data(train_file)
    # 这里是用分层划分, 但是由于细胞系的id是字符串型, 而sklearn中的划分方法要求传入数组, 字符串存不了数组, 因此需要用映射
    cell_line_id_mapping = {symbol: i for i, symbol in enumerate(pd.unique(feature['cell_line_name']))}
    feature['cell_line_name'].replace(cell_line_id_mapping, inplace=True)
    count = feature['cell_line_name'].value_counts()
    feature = feature.to_numpy()
    label = label.to_numpy()
    # train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.3, random_state=666)
    # feature: 药物一, 药物二, 细胞系
    return feature, label, cell_line_id_mapping


def load_cell_line_file_mapping(cell_line_dir):
    '''
    加载细胞系文件的映射字典
    '''
    cell_line_file_mapping = {}
    files = os.listdir(cell_line_dir)
    for file in files:
        symbol = file.split('.')[0]
        cell_line_file_mapping[symbol] = cell_line_dir + '/' + file
    return cell_line_file_mapping


def concat_ABdrug(input_data, cell_line_mapping):
    drugdim = 955
    pre_cell_line = None
    sample_nums = len(input_data)
    feature = np.zeros((sample_nums, drugdim + drugdim))

    for i, line in enumerate(input_data):
        cell_line = line[2]
        drug_1 = line[0]
        drug_2 = line[1]
        if cell_line != pre_cell_line:
            path = cell_line_mapping[cell_line][1]
            data = pd.read_csv(path)
        feature[i] = np.concatenate((data[str(drug_1)], data[str(drug_2)]))

    # feature_pd = pd.DataFrame(data=feature, index=[i for i in range(sample_nums)])
    # feature = torch.from_numpy(feature).float()
    return feature


def construct_NN_graph(dG):
    term_layer_list = []  # term_layer_list stores the built neural network # 建立层级和term之间的关系
    term_neighbor_map = {}

    # term_neighbor_map records all children of each term
    for term in dG.nodes():  # 建立字典：term_neighbor_map
        term_neighbor_map[term] = []
        for child in dG.neighbors(term):  # 为啥不用descendants()？？
            term_neighbor_map[term].append(child)
    wzm_i = 0
    layer_id = 0
    while True:
        leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]  # 出度为0的即为叶子节点，即为当前层的叶子节点

        wzm_i += 1
        print(wzm_i, 'len', len(leaves))
        if len(leaves) == 0:  # 如果没有叶子节点，则跳出循环
            break

        term_layer_list.append(leaves)  # 把一批叶子作为一层layer，作为term_layer_list中的一个元素，得到term和layer的关系
        dG.remove_nodes_from(leaves)  # 处理完当前层叶子节点之后，删除当前层叶子节点，继续下一轮

    return term_layer_list, term_neighbor_map


if __name__ == '__main__':
    dict1 = {'a': [10, 1], 'b': [1, 8], 'c': [1, 2]}
    dict2 = {'a': [1, 6], 'c': [1, 4]}
    a = dict2.update(dict1)
    a = merge_dict(dict1, dict2)
    cur_time = time.asctime(time.localtime(time.time()))
    print(cur_time)
    # pd.DataFrame(data = a).to_csv('./result/test.csv', mode='ab+')
    write_log_to_csv('./result/test.csv', cur_time)
    pd.DataFrame(dict1).to_csv('./result/test.csv', mode='ab+')

    print(a)

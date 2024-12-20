import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
from .evaluation import *

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def save_variable(save_dir, *arg, **args):
    """
    保存变量到文件夹
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i, var in enumerate(arg):
        if 'variable_name' in args:
            var_name = args['variable_name'][i]
        else:
             var_name = namestr(var, globals())[0]
        f_handle = open(os.path.join(save_dir, var_name), mode='wb')
        joblib.dump(var, f_handle)
        f_handle.close()

def get_term_score_dict_rr(y_true, val_predict_dict: dict, metric_name='r2'):
    """获取每个term对应得分的字典

    parameters
    ----------
        y_true: 训练集(验证集)的真实标签
        val_predict: 模型在验证集上的预测结果字典
        metric_name: 指标的名称, ['r2', 'pcc', 'spearman', 'mse', 'rmse', 'mae', 'medae', 'mape']
    return
    ------
        term_score_dict: dict
            key: term's name
            value: term's score
    """
    metric = eval(metric_name)
    term_score_dict = {}
    for term, val_pred in val_predict_dict.items():
        term_score_dict[term] =  metric(y_true, np.reshape(val_pred, (-1)))
    return term_score_dict

def get_term_score_dict_clf(y_true, val_predict_dict: dict, metric_name='f1'):
    """获取每个term对应得分的字典

    parameters
    ----------
        y_true: 训练集(验证集)的真实标签
        val_predict: 模型在验证集上的预测结果字典
        metric_name: 指标的名称, ['f1', 'acc', 'recall', 'roc']
    return
    ------
        term_score_dict: dict
            key: term's name
            value: term's score
    """
    metric = eval(metric_name)
    term_score_dict = {}
    for term, val_pred in val_predict_dict.items():
        term_score_dict[term] =  metric(y_true, np.reshape(val_pred, (-1)))
    return term_score_dict

def get_weights(distance_mat) -> np.ndarray:
    """
    distance_mat shape (sample number, -1)
    """
    reciprocal = 1 / (distance_mat / np.sum(distance_mat, axis=1).reshape((-1, 1)))
    weights = reciprocal / np.sum(reciprocal, axis=1).reshape((-1, 1))
    return weights

def get_dist(x, center) -> np.ndarray:
    # 默认使用欧氏距离
    if len(np.shape(x)) != len(np.shape(center)):
        center =  np.reshape(center, (1, -1))       # 如果是多个center怎么办?
    # dist_mat = european(x, center)
    dist_mat = cosine_distances(x, center)
    return dist_mat

def european(mat1, mat2) -> float:
    if check_shape(mat1, mat2):
        return np.reshape(np.sum(mat1 ** 2, axis=1), (mat1.shape[0], 1)) + np.sum(mat2 ** 2, axis=1) - 2 * mat1.dot(
            mat2.T)

def cosine_distances(mat1, mat2) -> float:
    if check_shape(mat1, mat2):
        return 1 - cosine_similarity(mat1, mat2)

def check_shape(mat1, mat2) -> bool:
    if isinstance(mat1, np.ndarray) & isinstance(mat2, np.ndarray):
        return True
    else:
        return False


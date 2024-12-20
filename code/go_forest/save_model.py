# 保存模型
import os

import xgboost as xgb
import  joblib
import numpy as np
from .node_add_rotation import NodeCLF
from .util import save_variable

def mkdir_model(f_path, term_layer_list):
    """
    parameters
    ----------
        f_path: str
            路径
        term_layer_list: list 

    """
    if not os.path.exists(f_path): os.mkdir(f_path)
    for i, layer in enumerate(term_layer_list):
        l_path = os.path.join(f_path, f"layer{i}")
        if not os.path.exists(l_path): os.mkdir(l_path)
        for term_name in layer:
            t_path = os.path.join(l_path, term_name)
            if not os.path.exists(t_path): os.mkdir(t_path)

# def save_xgbs(term_node: Node, f_path: str):
#     layer_id = term_node.layer_id
#     term_name = term_node.term_name
#     save_dir = os.path.join(f_path, f"layer{layer_id}/{term_name}")
#     if not os.path.exists(save_dir):
#         mkdir_model(f_path=f_path)
#     # save model
#     for i, est in enumerate(term_node.estimators):
#         est.save_model(os.path.join(save_dir, f"xgbr000{i}.json"))
    

def load_xgbs(path, forest_id):
    """加载xgbs模型
    parameters
    ----------
        path: str
            某个term存放model的文件夹
        forest_id: int
            该term_node中的森林的编号
    return
    ------
        model: xgboost
            xgboost的模型
    """
    model = xgb.XGBRegressor()
    xgb_path = os.path.join(path, f"xgbr000{forest_id}.json")
    model.load_model(xgb_path)
    return model

def load_train_input(path, forest_id=None):
    """加载训练输入数据
    parameters
    ----------
        path: str
            某个term存放model的文件夹
        forest_id: int
            该term_node中的森林的编号
    return 
    ------
        x_train
        y_train
    """
    if forest_id != None:
        f_path = os.path.join(path, 'trainall_input')
    else:
        f_path = os.path.join(path, f"train{forest_id}_input")
    with open(f_path, 'rb') as f_handle:
        input = joblib.load(f_handle)

    x_train, y_train = np.split(input, [-1], 1)
    return x_train, y_train
    
def load_feature_names(path):
    """加载特征名称
    parameters
    ----------
        path: str
            某个term存放model的文件夹
    """
    with open(os.path.join(path, "feature_names"), "rb") as f_handle:
        feature_names = joblib.load(f_handle)
    return feature_names

def load_term_layer_list():
    path = "/home/tq/project/drugcomb_V1009/data/cachedata/网络构建/term_layer_list"
    f_handle = open(path, mode='rb')
    term_layer_list = joblib.load(f_handle)
    f_handle.close()
    return term_layer_list

def load_term_direct_gene_map():
    path = "/home/tq/project/drugcomb_V1009/data/cachedata/网络构建/term_direct_gene_map"
    f_handle = open(path, mode='rb')
    term_direct_gene_map = joblib.load(f_handle)
    f_handle.close()
    return term_direct_gene_map


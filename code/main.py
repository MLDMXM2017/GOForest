
import sys
import time
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
from scipy.stats import spearmanr, pearsonr


from sklearn.model_selection import RepeatedKFold
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
print(BASE_DIR)

from go_forest.evaluation import scores_rr
from go_forest.goforest import GOForestRegressor
log_dir = '模型日志'
save_model = False

random_seed = 666
version = 'dc1'
base_path = F"../data/{version}"
model_names = ['GOForest']
metric_names = ['R2', 'PCC', 'SCC', 'MSE', 'RMSE', 'MAE']
n_fold = 5

def pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]
def spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]
def root_mse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ######################        函数          #######################
def recur_mkdir(path: str):
    """创建文件夹, 如果文件夹不存在, 就逐层创建父级文件夹, 直到最后一级文件夹被创建"""
    dirs = path.split("/")
    path = dirs[0]
    for dir in dirs[1:]:
        path = os.path.join(path, dir)
        if not os.path.exists(path):
            os.mkdir(path)

def get_result(x, y, cv_list):
    metric_func_list = [
        r2_score,
        pearson,
        spearman,
        mean_squared_error,
        root_mse,
        mean_absolute_error,
    ]

    results = []
    for i, (train_idx, test_idx) in enumerate(cv_list):
        print("---------------------------------------------------------")
        print(f"Fold {i + 1} start...")
        start = time.time()

        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        # 初始化模型
        goforest_configs = {
            'is_bypass_train': False,           # 训练阶段是否bypass
            'is_keep_exsample': False,          # 是否保留极端样本
            'is_bypass_predict': False,         # 预测阶段是否需要使用bypass
            'is_weighting_predict': True,      # 是否需要使用加权预测方式
            'poster_gene': 'neighbor',          # 使用哪些间接注释基因, 'neighbor', 'all'
            'poster_child': 'neighbor',         # 使用哪些孩子节点(增强向量), 'neighbor', 'all'
            'is_compress_lb_vector': True,          # 是否压缩层级的增强特征
            'lb_vector_compress_method': 'mean', # "linear_convert" or "mean"
            'lb_vector_compress_rate': 0.85,     # 压缩保留的信息量 
            'scaler': 50, 

        }
        layer_configs = {
            'is_weighting_predict': False,
        }
        node_configs = {
            'estimator': 'XGBRegressor()',
            'est_configs': {'n_estimators': 50, 'n_jobs': 10, 'objective': 'reg:squarederror'},
            'n_fold': 5,
            'n_atom': 1,
            'random_state': 666,
            'is_random_cv': True,
            'is_weighting_predict': True,
            'use_other_features': False,
            'is_rotating': False,
            'is_compress_boost_vec': True,                  # 是否压缩节点级增强向量
            'nb_vector_compress_method': 'mean',            # 压缩结点级增强向量的方法, 'mean' or 'linear_convert'
            'nb_vector_compress_rate': 0.85,               # 压缩nb时保留的信息量
            'is_transfer_lcb': True,                        # 是否传递层级的(压缩)增强特征
            'keep_origin_nb': True,                         # 是否保留原始结点级增强特征
            'log_dir': os.path.join(log_dir, 'log/中间结果'),
            'is_save_model': False,                           # 是否保存模型中间的结果
            'is_adaptive': False, 
        }
        recur_mkdir(node_configs['log_dir'])
        model = GOForestRegressor(
            term_layer_list, 
            term_direct_gene_map, 
            term_neighbour_map, 
            gene_dim, 
            root, 
            goforest_configs, 
            layer_configs,
            node_configs
            )
        with open(os.path.join(log_dir, "configs.txt"), mode='w') as f:
            f.write(f"{str(goforest_configs)}\n{str(layer_configs)}\n{str(node_configs)}")
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = np.reshape(y_pred, (-1))

        result = []
        for k in range(len(metric_names)):
            metric_func = metric_func_list[k]
            result.append(metric_func(y_test, y_pred))
        results.append([result])

        print(result)
        end = time.time()
        print(f"Fold {i + 1} end...")
        print(f"Time cost: {end-start}")
        print("---------------------------------------------------------")
    results = np.array(results)
    results_mean = np.mean(results, axis=0)
    results_std = np.std(results, axis=0)
    results = np.vstack((results, np.array([results_mean, results_std])))

    row_names = np.array([[''] + [f'fold {i + 1}' for i in range(n_fold)] + ['mean', 'std']]).T
    col_names = np.array([model_names])
    for k in range(len(metric_names)):
        metric_name = metric_names[k]
        result = results[:, :, k]
        print(row_names)
        print(col_names)
        print(result)
        np.savetxt(f"{base_path}/{version}_GOForest_{metric_name}.csv",
                   np.hstack((row_names, np.vstack((col_names, result)))),
                   fmt='%s', delimiter=',')

    return


if __name__ == '__main__':
    # ########################       第一部分 获取先验知识       ########################
    print("Model construction")
    cache_dir = "cachedata/ModelConstruction"
    cache_cv_file = "cachedata/cv.pkl"
    for root, _, file_list in os.walk(cache_dir):
        for file in file_list:
            f_handle = open(os.path.join(root, file), mode='rb')
            locals()[file] = joblib.load(f_handle)
            f_handle.close()
    term_layer_list = locals()["term_layer_list"]
    term_direct_gene_map = locals()["term_direct_gene_map"]
    term_neighbour_map = locals()["term_neighbour_map"]

    root = 'GO:0008150'
    gene_dim = 955
    feature_dim = 955*2

    # ###########################       第二部分        ##########################
    print("Data loading")
    x = pd.read_csv(f"{base_path}/Features.csv", index_col=0).iloc[:, 0:-1].values
    y = pd.read_csv(f"{base_path}/Labels.csv", index_col=0).iloc[:, 0].values
    cv_handle = open(f"{base_path}/cv.pkl", mode='rb')
    cv = joblib.load(cv_handle)
    cv_handle.close()

    print("Model training")
    get_result(x, y, cv)



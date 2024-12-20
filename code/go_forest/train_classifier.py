# GOForest 应用于DTI(分类任务)

import sys
import time
import os
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold, train_test_split
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.append(BASE_DIR)
from go_forest.goforest import GOForestClassifier
save_dir = '../ByPass训练+预测'
save_dir_result = '../ByPass训练+预测'
exp_name = 'ByPass训练+预测+50棵树'
save_model = False
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def recur_mkdir(path: str):
    """创建文件夹, 如果文件夹不存在, 就逐层创建父级文件夹, 直到最后一级文件夹被创建"""
    dirs = path.split("/")
    path = dirs[0]
    for dir in dirs[1:]:
        path = os.path.join(path, dir)
        if not os.path.exists(path):
            os.mkdir(path)
    
def scores_clf(y_true, y_proba):
    """
    return
    ------
        report: dict
    """
    y_pred = np.argmax(y_proba, axis=1)
    score_list = [
        roc_auc_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        accuracy_score(y_true ,y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
    ]
    score_name = [
        "roc_auc_score", 
        "f1_score",
        "accuracy_score",
        'precision_score',
        'recall_score',
    ]
    score_df = pd.DataFrame(score_list, index=score_name)
    return score_df

def convert_gene_id_to_feature_id(term_direct_gene_map, features):
    """将term_direct_gene_map中的geneid转化为gene在特征中的id, gene_map转化成0~954

    parameters
    ----------
        term_direct_gene_map: dict, 
            最原始的term_direct_gene_map
        features: DataFrame
            特征数据, columns必须是基因id
    return
    ------
        term_direct_gene_map: dict,
            映射之后的term_direct_gene_map
    """
    from copy import deepcopy
    term_direct_gene_map = deepcopy(term_direct_gene_map)
    gene2ind = features.columns[0:955]
    gene2ind_map = {}
    for index, feature_name in enumerate(gene2ind):
        gene = feature_name.split('_')[-1]
        gene2ind_map[gene] = index
    for term in term_direct_gene_map:
        for index, gene in enumerate(list(term_direct_gene_map[term])):
            term_direct_gene_map[term][index] = gene2ind_map[str(gene)] 

    return term_direct_gene_map

def train_model(
        term_layer_list,
        term_direct_gene_map,
        term_neighbour_map,
        gene_dim,
        root,
        goforest_configs,
        layer_configs,
        node_configs,
        x,
        y,
        phychem,
        fp
):
    with open(os.path.join(save_dir_model_log, "log/configs.txt"), mode='w') as f:
        f.write(f"{str(goforest_configs)}\n{str(layer_configs)}\n{str(node_configs)}")

    model = GOForestClassifier(
        term_layer_list, 
        term_direct_gene_map, 
        term_neighbour_map, 
        gene_dim, 
        root, 
        goforest_configs, 
        layer_configs,
        node_configs
        )

    model.fit(x, y, phychem, fp)
    return model


def train(save_dir, data_dirs, goforest_configs, layer_configs, node_configs, cv):

    cache_prior_dir = "/home/tq/project/先验知识/cachedata_网络构建"

    # ##########################       第一部分 获取先验知识, 特征标签数据       ##########################
    for root, _, file_list in os.walk(cache_prior_dir):
        for file in file_list:
            f_handle = open(os.path.join(root, file), mode='rb')
            locals()[file] = joblib.load(f_handle)
            f_handle.close()
    term_layer_list = locals()["term_layer_list"]
    term_neighbour_map = locals()["term_neighbour_map"]
    term_direct_gene_map = locals()["term_direct_gene_map"]
    term_direct_gene_map = convert_gene_id_to_feature_id(term_direct_gene_map, features_df)

    # ###########################       第二部分 训练&预测       ##########################
    print('this is the second part')

    ##########################        导入缓存数据       ############################
    with open(cache_cv_file, mode='rb') as cv_handle:   
        cv = joblib.load(cv_handle)
    root = term_layer_list[-1][0]
    # root = 'GO:0008150'
    gene_dim = 955
    feature_dim = 955*2

    phychem = None
    fp = None

    # 结果记录的变量
    metric_name_list = [
        "roc_auc_score", 
        "f1_score",
        "accuracy_score",
        'precision_score',
        'recall_score',
        ]
    
    train_score_df = pd.DataFrame(index=metric_name_list)
    train_average_score_df = pd.DataFrame(index=metric_name_list)
    test_score_df = pd.DataFrame(index=metric_name_list)
    test_average_score_df_1 = pd.DataFrame(index=metric_name_list)
    test_average_score_df_2 = pd.DataFrame(index=metric_name_list)
    test_root_score_df = pd.DataFrame(index=metric_name_list)

    
    for label_name, labels_se in label_df.iteritems():
        train_score, train_average_score, test_score, \
        test_average_score_1, test_average_score_2, test_root_score\
        = [], [], [], [], [], [] # 存储K折交叉验证的结果, 最多为K个元素. 
        
        labels = labels_se.values
        # 存储各个标签的中间结果
        save_dir_log = os.path.join(save_dir, label_name)
        recur_mkdir(save_dir_log)

        # 交叉验证
        loop = 0
        for train_index, test_index in cv:
            # if loop >= 1: break
            loop += 1
            save_dir_model_log = os.path.join(save_dir_log, f"cv{loop}")
            recur_mkdir(save_dir_model_log)
            # 整理训练数据
            if phychem is not None:
                phychem_train, fp_train = phychem[train_index], fp[train_index]
                phychem_test, fp_test = phychem[test_index], fp[test_index]
            else:
                phychem_train, fp_train = None, None
                phychem_test, fp_test = None, None
            x_train, y_train = features[train_index], labels[train_index]
            x_test, y_test = features[test_index], labels[test_index]

            # 默认的模型参数
            goforest_configs = {
                'is_bypass_train': False,           # 训练阶段是否bypass
                'is_keep_exsample': False,          # 是否保留极端样本
                'is_bypass_predict': False,         # 预测阶段是否需要使用bypass
                'is_weighting_predict': True,      # 是否需要使用加权预测方式
                'poster_gene': 'neighbor',          # 使用哪些间接注释基因, 'neighbor', 'all'
                'poster_child': 'neighbor',         # 使用哪些孩子节点(增强向量), 'neighbor', 'all'
                'is_compress_lb_vector': True,          # 是否压缩层级的增强特征
                'lb_vector_compress_method': 'mean', # "linear_convert" or "mean"
                'lb_vector_compress_rate': 0.9,   # 压缩保留的信息量 
                'scaler': 50,                      # 缩放参数, 默认50 

            }
            layer_configs = {
                'is_weighting_predict': False,
            }
            node_configs = {
                'estimator': 'XGBClassifier()',
                'est_configs': {'n_estimators': 50, 'n_jobs': 15, 'eval_metric':'logloss'},
                'n_fold': 5,
                'n_atom': 2,
                'random_state': 666,
                'is_random_cv': True,
                'is_weighting_predict': True,
                'use_other_features': False,
                'is_rotating': False,
                'is_compress_boost_vec': True,                  # 是否压缩节点级增强向量
                'nb_vector_compress_method': 'mean',        # 压缩结点级增强向量的方法, 'mean' or 'linear_convert'
                'nb_vector_compress_rate': 0.85,               # 压缩nb时保留的信息量
                'is_transfer_lcb': True,                        # 是否传递层级的(压缩)增强特征
                'keep_origin_nb': True,                         # 是否保留原始结点级增强特征
                'log_dir': os.path.join(save_dir_model_log, 'log/中间结果'),
                'is_save_model': False,                           # 是否保存模型中间的结果
                'is_adaptive': False, 
            }
            recur_mkdir(node_configs['log_dir'])

            # 训练模型
            model = train_model(
                term_layer_list,
                term_direct_gene_map,
                term_neighbour_map,
                gene_dim,
                root,
                goforest_configs,
                layer_configs,
                node_configs,
                x_train,
                y_train,
                phychem_train,
                fp_train
            )

            # 保存训练模型
            if save_model:
                m_handle = open(os.path.join(save_dir_model_log, "log/goforest.pkl"), mode='wb')
                joblib.dump(model, m_handle)
                m_handle.close()

            # # 加载训练好的模型
            # print("***************   load model   *****************")
            # m_handle = open(os.path.join(save_dir_model_log, "log/goforest.pkl"), mode='rb')
            # model = joblib.load(m_handle)
            # m_handle.close()

            # 使用模型预测训练集
            print("***************    predict train set    ****************")
            y_pred_proba_trian, _, _ = model.predict_proba(x_train, y_train, phychem_train, fp_train, )
            score_in_train = scores_clf(y_train, y_pred_proba_trian)
            train_score.append(score_in_train.values)

            y_pred_proba_train_average, _ = model.predict_average_proba()
            score_in_train_average = scores_clf(y_train, y_pred_proba_train_average)
            train_average_score.append(score_in_train_average.values)
            
            print("***************     predict in test set     ****************")
            # 使用模型预测测试集
            y_pred_proba_test, proba_mat_test, y_pred_proba_final_layer = model.predict_proba(x_test, y_test, phychem_test, fp_test)
            score_in_test = scores_clf(y_test, y_pred_proba_test)
            test_score.append(score_in_test.values)

            y_pred_proba_test_average_1, y_pred_proba_test_average_2 = model.predict_average_proba()
            score_in_test_average_1 = scores_clf(y_test, y_pred_proba_test_average_1)
            test_average_score_1.append(score_in_test_average_1.values)
            score_in_test_average_2 = scores_clf(y_test, y_pred_proba_test_average_2)
            test_average_score_2.append(score_in_test_average_2.values)

            score_in_test_root = scores_clf(y_test, y_pred_proba_final_layer)
            test_root_score.append(score_in_test_root.values)

            # 记录结果
            with open(os.path.join(save_dir_model_log, "record.txt"), "a") as f:
                f.write(f'{label_name}\n')
                for item in [score_in_train, score_in_train_average, score_in_test, 
                             score_in_test_average_1, score_in_test_average_2,
                             score_in_test_root]:
                    f.write(str(item)) 
                    f.write('\n')

            # 保存测试集预测的结果

            proba_mat_test = np.hstack(proba_mat_test)    # shape=(n_samples, n_terms*2)
            y_pred_test = np.argmax(y_pred_proba_test, axis=1)
            pred_model = pd.DataFrame(
                np.transpose([y_test, y_pred_test]), 
                columns=['y_true', 'weighted_predict']
            )
            pred_model[['class_0','class_1']] = y_pred_proba_test
            pred_model.to_csv(os.path.join(save_dir_model_log, "predict_values_model_测试集.csv"))

            y_pred_final_layer = np.argmax(y_pred_proba_final_layer, axis=1)
            pred_final_layer = pd.DataFrame(
                np.transpose([y_test, y_pred_final_layer]), 
                columns=['y_true', 'pred_final_layer']
            )
            pred_final_layer[['class_0','class_1']] = y_pred_proba_final_layer
            pred_final_layer.to_csv(os.path.join(save_dir_model_log, "predict_values_final_layer_测试集.csv"))

            y_pred_mean_1 = np.argmax(y_pred_proba_test_average_1)
            y_pred_mean_2 = np.argmax(y_pred_proba_test_average_2)
            pred_mean = pd.DataFrame(
                np.transpose([y_test, y_pred_mean_1, y_pred_mean_2]), 
                columns=['y_true', 'y_pred_mean_1', 'y_pred_mean_2']
            )
            pred_mean[['meth1_class_0','meth1_class_1']] =y_pred_proba_test_average_1
            pred_mean[['meth2_class_0','meth2_class_1']] =y_pred_proba_test_average_2
            pred_mean.to_csv(os.path.join(save_dir_model_log, "predict_values_mean_测试集.csv"))
            term_list = [term for item in term_layer_list for term in item]
            proba_class = pd.MultiIndex.from_product([term_list, ['class_0', 'class_1']], names=['term', 'class'])
            pd.DataFrame(proba_mat_test, columns=proba_class).to_csv(os.path.join(save_dir_model_log, "pred_proba_mat_测试集.csv"))

            del model

        # 计算平均值和方差
        # 平均值
        train_score_df[label_name] = np.mean(train_score, axis=0)
        train_average_score_df[label_name] = np.mean(train_average_score, axis=0)
        test_score_df[label_name] = np.mean(test_score, axis=0)
        test_average_score_df_1[label_name] = np.mean(test_average_score_1, axis=0)
        test_average_score_df_2[label_name] = np.mean(test_average_score_2, axis=0)
        test_root_score_df[label_name] = np.mean(test_root_score, axis=0)
        # 方差
        train_score_df[f"{label_name}_std"] = np.std(train_score, axis=0)
        train_average_score_df[f"{label_name}_std"] = np.std(train_average_score, axis=0)
        test_score_df[f"{label_name}_std"] = np.std(test_score, axis=0)
        test_average_score_df_1[f"{label_name}_std"] = np.std(test_average_score_1, axis=0)
        test_average_score_df_2[f"{label_name}_std"] = np.std(test_average_score_2, axis=0)
        test_root_score_df[f"{label_name}_std"] = np.std(test_root_score, axis=0)

    with pd.ExcelWriter(os.path.join(save_dir_result, f'result_{exp_name}.xlsx')) as writer:  
        train_score_df.to_excel(writer, sheet_name='train_score')
        train_average_score_df.to_excel(writer, sheet_name='train_average_score')
        test_score_df.to_excel(writer, sheet_name='test_score')
        test_average_score_df_1.to_excel(writer, sheet_name='test_average_score_1')
        test_average_score_df_2.to_excel(writer, sheet_name='test_average_score_2')
        test_root_score_df.to_excel(writer, sheet_name='test_root_score')

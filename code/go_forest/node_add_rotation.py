import copy
from multiprocessing import managers
import numpy as np
import pandas as pd
import os

from sklearn import pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble._forest import ForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from .util import get_dist, get_weights, save_variable
# from .cdutil import *
from .evaluation import *
from .search_params import warpper_search_parameters
from typing import List

class NodeRR:
    ROOT_NAME = 'GO:0008150'
    ROOT_LAYER_ID = 5
    RANDOM_SEED: int = 666
    TRAIN_STAGE: str = 'train'
    PREDICT_STAGE: str = 'predict'

    def __init__(
            self,
            term_name,
            layer_id,
            feature_idx,
            child_list,
            node_configs: dict,
    ):
        self.term_name = term_name
        self.layer_id = layer_id
        self.gene_idx = feature_idx
        self.child_list: list = child_list
        self.is_weighting_predict: bool = node_configs['is_weighting_predict']
        self.is_random_cv = node_configs['is_random_cv']
        self.use_other_features = node_configs['use_other_features']
        self.estimator = node_configs['estimator']
        self.est_configs = node_configs['est_configs']
        self.n_fold = node_configs['n_fold']
        self.n_atom = node_configs['n_atom']
        self.is_rotating = node_configs['is_rotating']
        self.is_compress_boost_vec = node_configs['is_compress_boost_vec']
        self.nb_vector_compress_method = node_configs.get('nb_vector_compress_method')    # 节点级压缩特征的方法, mean or linear_convert
        self.nb_vector_compress_reate = node_configs.get('nb_vector_compress_rate')         # 压缩nb时保留的信息量
        self.keep_origin_nb = node_configs.get('keep_origin_nb')            # 是否保留原始的结点级特征
        self.is_save_model = node_configs.get('is_save_model')
        self.is_transfer_lcb = node_configs['is_transfer_lcb']
        self.is_adaptive: bool = node_configs.get("is_adaptive")            # 是否要自适应搜索xgb的超参数, 将会耗费大量时间
        self.decomposition: int = 0 if node_configs.get("decomposition") is None else 1     # 线性降维方法选择, 如果是0, 则PCA, 如果是1, 则截断SVD
        self.log_dir = node_configs['log_dir'] 

        self.forest_centers = None
        self.node_center = None
        # initial basic estimators
        if (self.estimator is None) and (self.est_configs is None):
            self.estimators = [XGBRegressor(n_estimators=50, n_jobs=-1) for i in range(self.n_fold)]
        else:
            self.estimators = [eval(self.estimator).set_params(**self.est_configs) for i in range(self.n_fold)]
        # 创建缓存文件夹
        if self.is_save_model:
            self.mkdir_model()  


    def fit(self, x, y, train_boost_vec_dict=None, layer_cpboost_array=None, phychem=None, fp=None):
        if len(self.gene_idx) != 0:
            x_gene = x[:, self.gene_idx]
        else:
            x_gene = None
        x = self._process_features('train', x_gene, train_boost_vec_dict, layer_cpboost_array, phychem, fp)
        self._build_feature_names() # 构建特征名
        cv = self._get_cvs(x)
        # self._fit_estimators_multi_process(x, y, cv)  # 多进程会导致堵塞
        self._fit_estimators(x, y, cv)
        train_boost_vec = self._get_train_boost_vec(self.estimators, x, y, cv, self.n_atom)
        self.forest_centers = self._get_centers(x, cv=cv)
        self.node_center = self._get_centers(x, node_center=True)
        # print(self.term_name)
        return train_boost_vec

    def predict(self, x, pred_boost_vec_dict, layer_cpboost_array, phychem=None, fp=None):
        if len(self.gene_idx) != 0:
            x_gene = x[:, self.gene_idx]
        else:
            x_gene = None
        x_test = self._process_features('predict', x_gene, pred_boost_vec_dict, layer_cpboost_array, phychem, fp)
        # 先处理一维增强向量的结果
        y_pred_mat = []
        for est in self.estimators:
            _boost_vec = forestrr_predict(est, x_test, self.n_atom)    # shape: (len(x), n_atom)
            _y_pred = np.reshape(_boost_vec, -1)
            y_pred_mat.append(_y_pred)
        y_pred_mat = np.transpose(y_pred_mat)
        self.save_predict_input(x_test)
        if self.is_weighting_predict is False:
            # 不加权就 平均输出
            y_pred = np.mean(y_pred_mat, axis=1).reshape(-1)
        else:
            # 加权
            distance_mat = get_dist(x_test, self.forest_centers)
            weight_mat = get_weights(distance_mat)
            y_pred_list_adjusted = y_pred_mat * weight_mat
            y_pred_weighted = np.sum(y_pred_list_adjusted, axis=1)
            y_pred = y_pred_weighted.reshape(-1)
        pred_boost_vec = y_pred.reshape((-1, self.n_atom))

        return y_pred, pred_boost_vec

    def get_scores(self, x, y, pred_boost_vec_dict, phychem, fp) -> pd.DataFrame:
        y_pred, _ = self.predict_multi_process(x, pred_boost_vec_dict, phychem, fp)
        score_df = scores_rr(y, y_pred)
        return score_df

    def _compress_boost_vec(self, origin_boost_vec: np.ndarray, stage: str):
        """压缩增强特征
        
        return
        ------
            compressed_boost: ndarray
        """
        if origin_boost_vec.shape[1] > 1:    # 如果特征维度大于1维
            # 选择对应的压缩方式
            if self.nb_vector_compress_method == "linear_convert": 
                # 根据阶段的不同选择不同的操作(训练\预测)
                if stage == self.TRAIN_STAGE:
                    self.boost_compressor = pipeline.make_pipeline(
                        StandardScaler(),
                        PCA(n_components=self.nb_vector_compress_reate, svd_solver='full')
                    )

                    cp_boost = self.boost_compressor.fit_transform(origin_boost_vec)
                if stage == self.PREDICT_STAGE:
                    cp_boost = self.boost_compressor.transform(origin_boost_vec)
            if self.nb_vector_compress_method == "mean":
                cp_boost = np.mean(origin_boost_vec, axis=1).reshape((-1,1))
        elif origin_boost_vec.shape[1] == 1:
            cp_boost = origin_boost_vec
        return cp_boost

    def _add_boost_vec(self, stage, x, boost_vec_dict):
        """添加节点级增强特征(不处理层级)"""
        # 如果没有孩子(叶节点), 则直接返回基因特征
        if len(self.child_list) == 0:
            return x

        # 搜索原始nb(Node-level boost feature)
        origin_nb = None
        for term in self.child_list:
            if origin_nb is None:
                origin_nb = boost_vec_dict[term]
            else:
                origin_nb = np.hstack((origin_nb, boost_vec_dict[term]))
        # 压缩增强向量
        if self.is_compress_boost_vec:
            cp_boost = self._compress_boost_vec(origin_nb, stage)
            # 拼接
            if self.keep_origin_nb: 
                x_boost = np.hstack((origin_nb, cp_boost))  # 拼接原始增强特征和压缩的增强特征
            else:
                x_boost = cp_boost
            # 记录节点级压缩增强向量(ncb)的信息
            if stage == self.TRAIN_STAGE:
                self.ncb_train = cp_boost    # 记录训练阶段的节点级的压缩增强特征
            elif stage == self.PREDICT_STAGE:
                self.ncb_test = cp_boost     # 记录预测阶段的节点级的压缩增强特征

        else:   # 如果不压缩, 则保留全部结点级增强向量
            x_boost = origin_nb
            
        self.n_boost_features = x_boost.shape[1]
        if x is not None:
            x = np.hstack((x, x_boost))
        else:
            x = x_boost
        return x

    def _process_features(self, stage, x, boost_vec_dict, layer_cpboost_array, phychem=None, fp=None):
        x_gene = copy.deepcopy(x)
        # if self.layer_id != 0:  # 第0层不需要加增强向量
        # 添加物化性质和分子指纹特征 (仅针对根节点)
        if self.use_other_features and (self.layer_id == self.ROOT_LAYER_ID): 
            assert (phychem is None) | (fp is None), "检查是否要使用物化性质和分子指纹"
            if x_gene is not None:
                x = np.concatenate((x_gene, phychem, fp), axis=1)
            else: 
                x = np.concatenate((phychem, fp), axis=1)

        # 添加旋转的基因特征
        if x is not None and self.is_rotating and (self.layer_id != 5):   # 最后一层不需要添加旋转特征
            x_gene_rotated = self._rotate_gene(stage, x_gene)
            x = np.hstack([x, x_gene_rotated])
            self.rotate_dim = x_gene_rotated.shape[1]

        # 添加节点级的增强特征(Node-level boost features)
        if len(self.child_list) != 0 and self.keep_origin_nb:
            if x is not None:
                x = self._add_boost_vec(stage, x, boost_vec_dict)
            else:
                x = self._add_boost_vec(stage, None, boost_vec_dict)

        # 添加层级的增强特征(Layer-level boost features)
        if self.is_transfer_lcb and (len(layer_cpboost_array) != 0):
            self.lcb_dim = layer_cpboost_array.shape[1]  # 记录层级增强特征的维度
            if x is not None:
                x = np.hstack((x, layer_cpboost_array))      # 将cpboost_layer添加到x
            else:
                x = layer_cpboost_array
        return x

    def _rotate_gene(self, stage, x_gene):
        if stage == self.TRAIN_STAGE:
            self.rotater = pipeline.make_pipeline(
                StandardScaler(),
                TruncatedSVD(n_components=x_gene.shape[1]-1)
            )
            n_components = self.rotater.fit_transform(x_gene)
        elif stage == self.PREDICT_STAGE:
            n_components = self.rotater.transform(x_gene)
        return n_components


    def _fit_estimators(self, x, y, cv):
        # 搜索最优参数
        if self.is_adaptive:
            best_params, best_score = warpper_search_parameters(self, x, y, cv=cv)
            self.est_configs.update(best_params)
            self.best_params = copy.deepcopy(self.est_configs)
            # print(self.est_configs)
        # 训练学习器
        for i, est in enumerate(self.estimators):
            train_idx, val_idx = cv[i]
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
            # 为学习器设置参数
            est.set_params(**self.est_configs)
            est.fit(x_train, y_train)
            self.save_train_input(x_train, y_train, forest_id=i)    # 保存单个forest输入数据
        
        self.save_xgbs()
        self.save_feature_names()
        self.save_train_input(x_train, y_train)     # 保存整体的输入数据

    def _get_cvs(self, x: np.ndarray):
        random_seed = self.RANDOM_SEED if np.random.randint(0, 99999) else self.is_random_cv
        skf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        cv = [(t, v) for (t, v) in skf.split(x)]
        return cv

    def mkdir_model(self):
        layer_id = self.layer_id
        term_name = self.term_name
        f_path = self.log_dir.split('log')[0]
        model_dir = os.path.join(f_path, "model")
        layer_dir = os.path.join(f_path, f"model/layer{layer_id}")
        term_dir = os.path.join(f_path, f"model/layer{layer_id}/{term_name}")
        # make dictionary
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(layer_dir):
            os.mkdir(layer_dir)
        if not os.path.exists(term_dir):
            os.mkdir(term_dir)
        self.model_dir = term_dir
    def save_xgbs(self):
        """保存xgbr模型"""
        if not self.is_save_model: return
        save_dir = self.model_dir
        # save model
        for i, est in enumerate(self.estimators):
            est.save_model(os.path.join(save_dir, f"xgbr000{i}.json"))
    def save_train_input(self, x, y, forest_id='all'):
        """保存每个xgbr的输入"""
        if not self.is_save_model: return
        input = np.hstack((x, y.reshape((-1,1))))
        # if os.path.exists(self.model_dir+'/features_name'):
        #     save_variable(self.model_dir, self.feature_names, variable_name=['features_name'])
        save_variable(self.model_dir, input, variable_name=[f'train{forest_id}_input'])
    def save_predict_input(self, x):
        if not self.is_save_model: return
        save_variable(self.model_dir, x, variable_name=[f'predict_input'])  
    def _build_feature_names(self):
        # 添加原始基因特征
        self.feature_names = list(map(lambda idx: f"DrugA_gene{idx}" if idx < 955 else f"DrugB_gene{idx-955}", self.gene_idx))
        # 添加旋转基因特征
        if self.is_rotating:
                        self.feature_names.ex([f"rotation_feature_{i}" for i in range(self.rotate_dim)])
        # 添加原始的节点级增项特征
        if hasattr(self, 'child_list'):
            self.feature_names.extend(self.child_list)  
        # 添加压缩后的节点级增项特征
        if hasattr(self, 'ncb_train'):
            self.feature_names.extend([f"compress_node_level_boost_feature_{i}" for i in range(self.ncb_train.shape[1])])
        # 添加层级的增强特征名称
        if hasattr(self, 'lcb_dim'):
            self.feature_names.extend([f"compress_LAYER_level_boost_feature_{i}" for i in range(self.lcb_dim)])
        return self.feature_names
    def save_feature_names(self):
        """保存特征名称"""
        if not self.is_save_model: return
        # assert hasattr(self, 'feature_names')
        feature_names = self._build_feature_names()
        # 缓存feature_names
        save_variable(self.model_dir, self.feature_names, variable_name=['feature_names'])

    @staticmethod
    def _add_other_features(x, phychem, fp):
        x = np.hstack((x, phychem, fp))
        print(f"加上其他特征后的x的维度为 {x.shape[1]}")
        return x

    @staticmethod
    def _get_centers(x_train, **kwargs) -> np.ndarray:
        if len(x_train.shape) == 1:
            x_train = x_train.reshape((1, -1))
        centers = []
        if 'cv' in kwargs:
            cv: list = kwargs['cv']
            for train_idx, _ in cv:
                cen = np.mean(x_train[train_idx, :], axis=0)
                centers.append(cen)
            return np.array(centers)
        if 'node_center' in kwargs:
            node_center = np.mean(x_train, axis=0)
            return node_center

    @staticmethod
    def _get_train_boost_vec(estimators, x, y, cv, n_atom):
        train_boost_vec = np.zeros((len(x), n_atom))
        for i, est in enumerate(estimators):
            _, val_idx = cv[i]
            x_val = x[val_idx]
            y_pred_val = forestrr_predict(est, x_val, n_atom) # cdutil.py 和 node,py都有这个函数
            train_boost_vec[val_idx, :] = y_pred_val
        return train_boost_vec

class NodeCLF:
    ROOT_NAME = 'GO:0008150'
    ROOT_LAYER_ID = 5
    RANDOM_SEED: int = 666
    TRAIN_STAGE: str = 'train'
    PREDICT_STAGE: str = 'predict'

    def __init__(
            self,
            term_name,
            layer_id,
            feature_idx,
            child_list,
            node_configs: dict,
    ):
        self.term_name = term_name
        self.layer_id = layer_id
        self.gene_idx = feature_idx
        self.child_list: list = child_list
        self.is_weighting_predict: bool = node_configs['is_weighting_predict']
        self.is_random_cv = node_configs['is_random_cv']
        self.use_other_features = node_configs['use_other_features']
        self.estimator = node_configs['estimator']
        self.est_configs = copy.deepcopy(node_configs['est_configs'])                   # 深拷贝是因为后续est_configs可能会被改变
        self.n_fold = node_configs['n_fold']
        self.n_atom = node_configs['n_atom']
        self.is_rotating = node_configs['is_rotating']
        self.is_compress_boost_vec = node_configs['is_compress_boost_vec']
        self.nb_vector_compress_method = node_configs.get('nb_vector_compress_method')    # 节点级压缩特征的方法, mean or linear_convert
        self.nb_vector_compress_reate = node_configs.get('nb_vector_compress_rate')         # 压缩nb时保留的信息量
        self.keep_origin_nb = node_configs.get('keep_origin_nb')            # 是否保留原始的结点级特征
        self.is_save_model = node_configs.get('is_save_model')
        self.is_transfer_lcb = node_configs['is_transfer_lcb']
        self.is_adaptive: bool = node_configs.get("is_adaptive")            # 是否要自适应搜索xgb的超参数, 将会耗费大量时间
        self.decomposition: int = 0 if node_configs.get("decomposition") is None else 1
        self.log_dir = node_configs['log_dir']

        self.forest_centers = None
        self.node_center = None
        # initial basic estimators
        if (self.estimator is None) and (self.est_configs is None):
            self.estimators = [XGBClassifier(n_estimators=50, n_jobs=-1) for i in range(self.n_fold)]
        else:
            self.estimators = [eval(self.estimator).set_params(**self.est_configs) for i in range(self.n_fold)]
        # 创建缓存文件夹
        if self.is_save_model:
            self.mkdir_model()  

    def fit(self, x, y, train_boost_vec_dict=None, layer_cpboost_array=None, phychem=None, fp=None):
        """
        Returns
        -------
        nob_out: ndarray, shape=(n_sample, n_atom).
        y_pred_proba: ndarray, class probability in validation set, shape=(n_sample, n_class).
        """
        self.n_class = len(np.unique(y))
        if len(self.gene_idx) != 0:
            x_gene = x[:, self.gene_idx]
        else:
            x_gene = None
        x = self._process_features('train', x_gene, train_boost_vec_dict, layer_cpboost_array, phychem, fp)
        self._build_feature_names() # 构建特征名
        cv = self._get_cvs(x)
        self._fit_estimators(x, y, cv)
        nob_out, y_pred_proba = self._get_train_boost_vec(self.estimators, x, y, cv, self.n_atom, self.n_class)
        self.forest_centers = self._get_centers(x, cv=cv)
        self.node_center = self._get_centers(x, node_center=True)
        # print(self.term_name)
        return nob_out, y_pred_proba

    def predict(self, x, pred_boost_vec_dict, layer_cpboost_array, phychem=None, fp=None):
        pred_proba, boost_vec = self.predict_proba(x, pred_boost_vec_dict, layer_cpboost_array, phychem, fp)
        y_pred = np.argmax(pred_proba, axis=1)
        return y_pred, boost_vec

    def predict_proba(self, x, pred_boost_vec_dict, layer_cpboost_array, phychem=None, fp=None):
        """预测概率
        
        Returns
        ------
        y_pred_proba: 结点的预测类概率
        nob_out: 结点级原始增强向量
        """
        if len(self.gene_idx) != 0:
            x_gene = x[:, self.gene_idx]
        else:
            x_gene = None
        x_test = self._process_features('predict', x_gene, pred_boost_vec_dict, layer_cpboost_array, phychem, fp)
        # 先处理一维增强向量的结果
        proba_mat = []  # shape: (n_est, n_samples)
        for est in self.estimators:
            _boost_vec = forestclf_predict(est, x_test)    # shape: (len(x), n_atom)
            proba_mat.append(_boost_vec)
        self.save_predict_input(x_test)
        if self.is_weighting_predict is False:
            # 不加权就 平均输出
            y_pred_proba = np.mean(proba_mat, axis=0)
        else:
            # 加权
            distance_mat = get_dist(x_test, self.forest_centers)
            weight_mat = get_weights(distance_mat)
            weight_mat =np.transpose(weight_mat)[:, :, np.newaxis]
            y_pred_proba = np.sum(proba_mat * weight_mat, axis=0)     # shape: (n_smaples, 5)
        nob_out = self._get_predict_boost_vec(y_pred_proba, self.n_atom)

        return y_pred_proba, nob_out

    def get_scores(self, x, y, pred_boost_vec_dict, phychem, fp) -> pd.DataFrame:
        y_pred, _ = self.predict_multi_process(x, pred_boost_vec_dict, phychem, fp)
        score_df = scores_rr(y, y_pred)
        return score_df

    def _compress_boost_vec(self, origin_boost_vec: np.ndarray, stage: str):
        """压缩增强特征
        
        return
        ------
            compressed_boost: ndarray
        """
        if origin_boost_vec.shape[1] > 1:    # 如果特征维度大于1维
            # 选择对应的压缩方式
            if self.nb_vector_compress_method == "linear_convert": 
                # 根据阶段的不同选择不同的操作(训练\预测)
                if stage == self.TRAIN_STAGE:
                    self.boost_compressor = pipeline.make_pipeline(
                        StandardScaler(),
                        PCA(n_components=self.nb_vector_compress_reate, svd_solver='full')
                    )
                    cp_boost = self.boost_compressor.fit_transform(origin_boost_vec)
                if stage == self.PREDICT_STAGE:
                    cp_boost = self.boost_compressor.transform(origin_boost_vec)
            if self.nb_vector_compress_method == "mean":
                even_index = np.arange(0, np.shape(origin_boost_vec)[1], 2, dtype=int)   # 0类概率索引
                odd_index = np.arange(1, np.shape(origin_boost_vec)[1], 2, dtype=int)    # 1类概率索引
                mean_proba0 = np.mean(origin_boost_vec[:, even_index], axis=1).reshape((-1, 1))
                mean_proba1 = np.mean(origin_boost_vec[:, odd_index], axis=1).reshape((-1, 1))
                cp_boost = np.hstack((mean_proba0, mean_proba1))

        elif origin_boost_vec.shape[1] == 1:
            cp_boost = origin_boost_vec
        return cp_boost

    def _add_boost_vec(self, stage, x, boost_vec_dict):

        """添加节点级增强特征(不处理层级)"""
        # 如果没有孩子(叶节点), 则直接返回基因特征
        if len(self.child_list) == 0:
            return x

        # 搜索原始nb(Node-level boost feature)
        origin_nb = None
        for term in self.child_list:
            if origin_nb is None:
                origin_nb = boost_vec_dict[term]
            else:
                origin_nb = np.hstack((origin_nb, boost_vec_dict[term]))
        # 压缩增强向量
        if self.is_compress_boost_vec:
            cp_boost = self._compress_boost_vec(origin_nb, stage)
            # 拼接
            if self.keep_origin_nb: 
                x_boost = np.hstack((origin_nb, cp_boost))  # 拼接原始增强特征和压缩的增强特征
            else:
                x_boost = cp_boost
            # 记录节点级压缩增强向量（ncb)的信息
            if stage == self.TRAIN_STAGE:
                self.ncb_train = cp_boost    # 记录训练阶段的节点级的压缩增强特征
            elif stage == self.PREDICT_STAGE:
                self.ncb_test = cp_boost     # 记录预测阶段的节点级的压缩增强特征
        
        else:   # 如果不压缩, 则保留全部结点级增强向量
            x_boost = origin_nb
        
        # 记录节点级增强向量的信息
        self.n_boost_features = x_boost.shape[1]
        if x is not None:
            x = np.hstack((x, x_boost))
        else:
            x = x_boost
        return x

    def _process_features(self, stage, x, boost_vec_dict, layer_cpboost_array, phychem=None, fp=None):
        x_gene = copy.deepcopy(x)
        # if self.layer_id != 0:  # 第0层不需要加增强向量
        # 添加物化性质和分子指纹特征 (仅针对根节点)
        if self.use_other_features and (self.layer_id == self.ROOT_LAYER_ID): 
            assert (phychem is None) | (fp is None), "检查是否要使用物化性质和分子指纹"
            if x_gene is not None:
                x = np.concatenate((x_gene, phychem, fp), axis=1)
            else: 
                x = np.concatenate((phychem, fp), axis=1)

        # 添加旋转的基因特征
        if x is not None and self.is_rotating and (self.layer_id != 5):   # 最后一层不需要添加旋转特征
            x_gene_rotated = self._rotate_gene(stage, x_gene)
            x = np.hstack([x, x_gene_rotated])
            self.rotate_dim = x_gene_rotated.shape[1]

        # 添加节点级的增强特征(利用nob生成ncb)
        if len(self.child_list) != 0 and self.keep_origin_nb:
            if x is not None:
                x = self._add_boost_vec(stage, x, boost_vec_dict)
            else:
                x = self._add_boost_vec(stage, None, boost_vec_dict)

        # 添加层级的增强特征(Layer-level boost features)
        if self.is_transfer_lcb and (len(layer_cpboost_array) != 0):
            self.lcb_dim = layer_cpboost_array.shape[1]  # 记录层级增强特征的维度
            if x is not None:
                x = np.hstack((x, layer_cpboost_array))      # 将cpboost_layer添加到x
            else:
                x = layer_cpboost_array
        return x

    def _rotate_gene(self, stage, x_gene):
        if stage == self.TRAIN_STAGE:
            self.rotater = pipeline.make_pipeline(
                StandardScaler(),
                TruncatedSVD(n_components=x_gene.shape[1]-1)
            )
            n_components = self.rotater.fit_transform(x_gene)
        elif stage == self.PREDICT_STAGE:
            n_components = self.rotater.transform(x_gene)
        return n_components

    def _fit_estimators(self, x, y, cv):
        # 搜索最优参数
        if self.is_adaptive:
            best_params, best_score = warpper_search_parameters(self, x, y, scorer='accuracy')
            self.est_configs.update(best_params)
            self.best_params = copy.deepcopy(self.est_configs)
            # print(self.est_configs)
        # 训练学习器
        for i, est in enumerate(self.estimators):
            train_idx, val_idx = cv[i]
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
            # 为学习器设置参数
            est.set_params(**self.est_configs)
            est.fit(x_train, y_train)
            self.save_train_input(x_train, y_train, forest_id=i)    # 保存单个forest输入数据
        self.save_xgbs()
        self.save_feature_names()
        self.save_train_input(x_train, y_train)     # 保存整体的输入数据

    def _get_cvs(self, x: np.ndarray):
        random_seed = self.RANDOM_SEED if np.random.randint(0, 99999) else self.is_random_cv
        skf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        cv = [(t, v) for (t, v) in skf.split(x)]
        return cv

    def mkdir_model(self):
        """创建结点对应的文件夹"""
        layer_id = self.layer_id
        term_name = self.term_name
        f_path = self.log_dir.split('log')[0]
        model_dir = os.path.join(f_path, "model")
        layer_dir = os.path.join(f_path, f"model/layer{layer_id}")
        term_dir = os.path.join(f_path, f"model/layer{layer_id}/{term_name}")
        # make dictionary
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(layer_dir):
            os.mkdir(layer_dir)
        if not os.path.exists(term_dir):
            os.mkdir(term_dir)
        self.model_dir = term_dir

    def save_xgbs(self):
        """保存xgbr模型"""
        if not self.is_save_model: return
        save_dir = self.model_dir
        # save model
        for i, est in enumerate(self.estimators):
            est.save_model(os.path.join(save_dir, f"xgbr000{i}.json"))

    def save_train_input(self, x, y, forest_id='all'):
        """保存每个xgbr的输入"""
        if not self.is_save_model: return
        input = np.hstack((x, y.reshape((-1,1))))
        # if os.path.exists(self.model_dir+'/features_name'):
        #     save_variable(self.model_dir, self.feature_names, variable_name=['features_name'])
        save_variable(self.model_dir, input, variable_name=[f'train{forest_id}_input'])
    
    def save_predict_input(self, x):
        if not self.is_save_model: return
        save_variable(self.model_dir, x, variable_name=[f'predict_input'])  
    
    def save_feature_names(self):
        """保存特征名称"""
        if not self.is_save_model: return
        # assert hasattr(self, 'feature_names')
        feature_names = self._build_feature_names()
        # 缓存feature_names
        save_variable(self.model_dir, self.feature_names, variable_name=['feature_names'])
    
    def _build_feature_names(self):
        # 添加原始基因特征
        self.feature_names = list(map(lambda idx: f"DrugA_gene{idx}" if idx < 955 else f"DrugB_gene{idx-955}", self.gene_idx))
        # 添加旋转基因特征
        if self.is_rotating:
                        self.feature_names.ex([f"rotation_feature_{i}" for i in range(self.rotate_dim)])
        # 添加原始的节点级增项特征
        if hasattr(self, 'child_list'):
            for term_name in self.child_list:
                self.feature_names.extend([f"{term_name}_class0", f"{term_name}_class1"])
        # 添加压缩后的节点级增项特征
        if hasattr(self, 'ncb_train'):
            self.feature_names.extend([f"ncb_{i}" for i in range(self.ncb_train.shape[1])])
        # 添加层级的增强特征名称
        if hasattr(self, 'lcb_dim'):
            self.feature_names.extend([f"lcb{i}" for i in range(self.lcb_dim)])
        return self.feature_names

    @staticmethod
    def _add_other_features(x, phychem, fp):
        x = np.hstack((x, phychem, fp))
        print(f"加上其他特征后的x的维度为 {x.shape[1]}")
        return x

    @staticmethod
    def _get_centers(x_train, **kwargs) -> np.ndarray:
        if len(x_train.shape) == 1:
            x_train = x_train.reshape((1, -1))
        centers = []
        if 'cv' in kwargs:
            cv: list = kwargs['cv']
            for train_idx, _ in cv:
                cen = np.mean(x_train[train_idx, :], axis=0)
                centers.append(cen)
            return np.array(centers)
        if 'node_center' in kwargs:
            node_center = np.mean(x_train, axis=0)
            return node_center

    @staticmethod
    def _get_train_boost_vec(estimators, x, y, cv, n_atom, n_class):
        """获取训练阶段的原始增强特征"""
        train_proba = np.zeros((len(x), n_class))
        for i, est in enumerate(estimators):
            _, val_idx = cv[i]
            x_val = x[val_idx]
            y_pred_val = forestclf_predict(est, x_val)
            train_proba[val_idx, :] = y_pred_val
        # 如果限定增强特征长度为1, 则转为预测结果
        if n_atom == 1:
            train_boost_vec = np.argmax(train_proba, axis=1).reshape((-1, 1))
        else:
            train_boost_vec = train_proba
        return train_boost_vec, train_proba

    @staticmethod
    def _get_predict_boost_vec(proba, n_atom):
        """
        获取预测阶段的原始增强特征
        输入: 预测概率
        输出: 增强特征
        """
        if n_atom == 1:
            boost_vec = np.argmax(proba, axis=1).reshape((-1, 1))
        else:
            boost_vec = proba
        return boost_vec

def forestrr_predict(forest, x: np.ndarray, n_atom: int):
    """
    重新处理森林的预测方式：森林的输出是所有该森林中所有决策树的输出
    n_atom: 输出的增强向量的维度
    """
    if x.shape[1] != forest.n_features_in_:
        print("the shape of x not equal the n_features_in_ of the passed forest")
        return
    if isinstance(forest, ForestRegressor):
        _estimators = forest.estimators_.copy()
        n_estimators = len(_estimators)
        step = int(n_estimators / n_atom)
        y_pred = np.array([[]] * x.shape[0])
        for i, tree in enumerate(_estimators):
            y_pred = np.hstack((y_pred, np.reshape(tree.predict(x), (-1, 1))))
        if step != 1:
            for i in range(n_atom):
                y_pred[:, i] = np.mean(y_pred[:, step * i:step * (i + 1)], axis=1)
    else:
        assert n_atom == 1, "当前的基学习器不支持n_atom大于1的设置"
        y_pred = forest.predict(x).reshape((-1, 1))
    
    return y_pred[:, 0:n_atom]

def forestclf_predict(forest, x: np.ndarray):
    """
    分类器森林的预测方式, 重新处理森林的预测方式：森林的输出是所有该森林中所有决策树的输出
    n_atom: 输出的增强向量的维度
    """
    assert x.shape[1] == forest.n_features_in_, "the shape of x not equal the n_features_in_ of the passed forest"
    pred = forest.predict_proba(x)

    return pred




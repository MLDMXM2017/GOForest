import os
import numpy as np
from operator import itemgetter
from copy import deepcopy
import joblib
import pandas as pd
from .layer import LayerRR, LayerCLF
from .weighter import RRWeighter, CLFWeighter
from .util import get_dist, get_term_score_dict_clf, get_term_score_dict_rr, get_weights, namestr, save_variable
from typing import List
import logging

class GOForestRegressor:
    EXTRA_THRESHOLD: int = 15
    CUDA_ID: int = 3
    RECON_ERROR_THRESHOLD: float = 0.25

    def __init__(
            self,
            term_layer_list: list,
            term_direct_gene_map: dict,  # 假设value是list, 而不是ndarray
            term_neighbour_map: dict,
            gene_dim: int,  # gene维度, 955
            root: str,
            forest_configs: dict,
            layer_configs: dict,
            node_configs=None,
            **kwargs,
    ):
        """
        kwargs: 
            term_child_map
        """
        if kwargs:
            self.kwargs = kwargs
            self.term_child_map = self.kwargs.get("term_child_map")
        self.log_dir = node_configs['log_dir']
        self.log_dir = node_configs['log_dir']
        self.root = root
        self.gene_dim = gene_dim
        self.n_layers = len(term_layer_list)
        self.term_layer_list = term_layer_list
        self.term_neighbor_map = term_neighbour_map

        # self.is_bypass_train = forest_configs['is_bypass_train']
        # self.is_keep_exsample = forest_configs['is_keep_exsample']
        # self.is_bypass_predict = forest_configs['is_bypass_predict']
        self.is_bypass_train = False
        self.is_keep_exsample = False
        self.is_bypass_predict = False
        self.is_weighting_predict = forest_configs['is_weighting_predict']
        self.poster_gene = forest_configs['poster_gene']      # 'neighbor' or 'all' or 'None'
        self.poster_child = forest_configs['poster_child']      # 'neighbor' or 'all' or 'None'
        self.is_compress_lb_vector = forest_configs["is_compress_lb_vector"]
        self.lb_vector_compress_rate = forest_configs["lb_vector_compress_rate"]
        self.lb_vector_compress_method = forest_configs["lb_vector_compress_method"]
        self.scaler = forest_configs.get("scaler")          # 用于缩放权重的参数, 默认50

        self.term_layer_map = self._init_term_layer_map()
        self.term_direct_gene_map = self._pair_gene(term_direct_gene_map)
        self.neighbor_layer_list = self._get_neighbor_layer_list()
        self.child_layer_list = self._get_child_layer_list(self.poster_child)
        self.gene_layer_list = self._get_gene_layer_list_with_poster(self.poster_gene)

        self.layers: List[LayerRR] = self._init_layer(layer_configs, node_configs)

    def _get_est_params(self):
        """获取node中基学习器的参数"""
        params_list = []
        for term_name, layer_id in self.term_layer_map.items():
            node = self._get_node(term_name)
            params_se = pd.Series(node.est_configs)
            params = params_se.values
            params = np.append([term_name, layer_id], params_se.values)
            params_list.append(params)
        columns = np.append(["term's name", "layer id"], params_se.index)
        
        pd.DataFrame(params_list, columns=columns).to_csv(os.path.join(self.log_dir, "xgbs_params.csv"))

        
    def _init_term_layer_map(self):
        """初始化term_layer映射的字典, 键值对为term_name: layer_id
        """
        term_layer_map = {}
        for layer_id, layer in enumerate(self.term_layer_list):
            for term_name in layer:
                term_layer_map[term_name] = layer_id
        return term_layer_map

    def _get_node(self, term_name: str):
        """根据term_name获取node对象
        
        parameters
        ----------
            term_name: str, term's name
            
        return
        ------
            node: Node, term_name对应的Node对象
        """
        layer_id = self.term_layer_map[term_name]
        return self.layers[layer_id].node_dict[term_name]

    def _init_layer(self, layer_configs, node_configs):
        layer_list = []
        # gene_layer_list = self._get_gene_layer_list_with_poster()
        # child_layer_list = self._get_child_layer_list()
        for layer_id in range(self.n_layers):
            l = LayerRR(
                layer_id,
                self.term_layer_list[layer_id],
                self.gene_layer_list[layer_id],
                self.child_layer_list[layer_id],
                layer_configs,
                node_configs
            )
            layer_list.append(l)
        return layer_list
    
    def fit(self, x, y, phychem=None, fp=None):
        x_origin = deepcopy(x)
        y_origin = deepcopy(y)
        full_idx = np.arange(len(x_origin))
        hard_idx = np.arange(len(x_origin))
        hard_idx_list = []
        easy_idx_list = []
        boost_vec_dict = {}         # 训练阶段的增强向量字典
        layer_cpboost_dict = {}     # 训练阶段的cpboost_layer的字典
        for layer_id, layer in enumerate(self.layers):
            x, y = x_origin[hard_idx], y_origin[hard_idx]
            adjusted_boost_vec_dict = self._draw_boost_vec_dict(boost_vec_dict, hard_idx, full_idx)
            # layer_cpboost = self._draw_boost_vec_dict(layer_cpboost_dict, hard_idx, full_idx)
            if phychem is not None: 
                phychem = phychem[hard_idx]
            if fp is not None:
                fp = fp[hard_idx]

            if layer_id < self.n_layers-1:
                y_pred_layer, boost_vec_layer = layer.fit(
                    x, y,
                    boost_vec_dict_in=adjusted_boost_vec_dict,
                    layer_cpboost_dict=layer_cpboost_dict,
                )
                if self.is_compress_lb_vector:
                    self._update_lcb_vector(layer_id, layer_cpboost_dict, 'train')
            else:
                y_pred_layer, boost_vec_layer = layer.fit(
                    x, y,
                    boost_vec_dict_in=adjusted_boost_vec_dict,
                    layer_cpboost_dict=layer_cpboost_dict,
                    phychem=phychem, fp=fp
                )
            boost_vec_layer = self._embed_boost_vec_dict(boost_vec_layer, hard_idx, full_idx)
            boost_vec_dict.update(boost_vec_layer)

            if self.is_bypass_train:
                hard_idx, easy_idx = self.bypass_train(y_origin, y_pred_layer, hard_idx)
                easy_idx_list.append(easy_idx)
                hard_idx_list.append(hard_idx)
            print(f"训练完第{layer_id}层")
        # 如果加权
        if self.is_weighting_predict:
            self.weighter = self._init_weighter(y_origin, boost_vec_dict)
        
        # 保存训练中间结果
        save_variable(self.log_dir, boost_vec_dict, variable_name=['boost_vec_dict'])
        save_variable(self.log_dir, layer_cpboost_dict, variable_name=['train_layer_cpboost_dict'])
        self._get_boost_vec_number()
        self._get_cpboost_length(layer_cpboost_dict)
        self._get_est_params()
        return boost_vec_dict, easy_idx_list, hard_idx_list

    def predict(self, x, phychem=None, fp=None):
        x_origin = deepcopy(x)
        full_idx = np.arange(len(x_origin))
        hard_idx = np.arange(len(x_origin))
        easy_idx = []
        hard_idx_list = []
        easy_idx_list = []
        boost_vec_dict = {}         # 预测阶段的 增强向量(其实是节点的预测输出)字典
        layer_cpboost_dict = {}     # 预测阶段的 cpboost_layer 字典
        y_pred = np.zeros(len(x))
        y_pred_mat = [] # initial shape: (n_layers, sample_num)
        for layer_id, layer in enumerate(self.layers):
            x = x_origin[hard_idx]
            adjusted_boost_vec = self._draw_boost_vec_dict(boost_vec_dict, hard_idx, full_idx)
            boost_vec_layer = {}
            if layer_id < self.n_layers-1:
                layer_pred, boost_vec_layer = layer.predict(x, adjusted_boost_vec, layer_cpboost_dict)
                if self.is_compress_lb_vector:
                    self._update_lcb_vector(layer_id, layer_cpboost_dict, 'predict')
            else:
                layer_pred, boost_vec_layer = layer.predict(x, adjusted_boost_vec, layer_cpboost_dict, phychem, fp)
            y_pred[hard_idx] = layer_pred
            y_pred_mat.append(deepcopy(y_pred))
            boost_vec_layer = self._embed_boost_vec_dict(boost_vec_layer, hard_idx, full_idx)
            boost_vec_dict.update(boost_vec_layer)

            if self.is_bypass_predict:
                hard_idx, easy_idx = self.bypass_predict(x_origin, x, layer_id)
                hard_idx_list.append(hard_idx)
                easy_idx_list.append(easy_idx)

        y_pred_mat = np.transpose(y_pred_mat)   # now shape: (sample_num, n_layers)
        final_layer_predict = y_pred
        # 加权预测
        if self.is_weighting_predict:
            y_pred = self.weighter.weight_predict(boost_vec_dict)
            save_variable(self.log_dir, self.weighter.weights, y_pred, variable_name=['nodes_weight', 'weighted_pred'])
            self.weighter.weights.to_csv(os.path.join(self.log_dir, "结点weights.csv"))

        # 保存中间预测结果
        self.y_pred_mat = y_pred_mat
        save_variable(self.log_dir, final_layer_predict, y_pred_mat, boost_vec_dict, variable_name=['final_layer_predict', 'y_pred_mat', 'boost_vec_dict_predict_stage'])
        # 
        self.nob_dict = boost_vec_dict
        self.lcb_dict = layer_cpboost_dict
        return y_pred

    def predict_average(self):
        """
        返回各层的平均预测值
        """
        if self.y_pred_mat is not None:
            y_pred_average = np.mean(self.y_pred_mat, axis=1)
            save_variable(self.log_dir, y_pred_average, variable_name=['y_pred_mean_layer'])
            return y_pred_average.reshape(-1)

    def predict_block(self, x, drop_list: List[List]):
        """屏蔽样本的重要子系统再做预测
        Parameters
        ----------
        x: samples, 
        drop_list: list, shape=(n_samples, )
            drop_list 中的每一个元素是list, 存储了每个样本对应的重要结点
        """
        # build drop_mask
        term_list = list(self.weighter.individual_names)
        drop_mask = np.ones((len(x), len(term_list)))
        for i, terms in enumerate(drop_list):
            for term in terms:
                j = term_list.index(term)
                drop_mask[i, j] = 0
       # predict
        self.predict(x, None, None)
        y_pred_dict = self.nob_dict
        term_weights = np.array(self.weighter.weights).squeeze()   # shape=(n_terms, )
        weight_mat = term_weights * drop_mask
        weight_mat = weight_mat / np.sum(weight_mat, axis=1).reshape((-1, 1))
        y_pred_mat = self.weighter._processing_y_predict_dict(y_pred_dict)    # shape=(n_samples, n_terms, )
        y_pred = np.sum(weight_mat * y_pred_mat, axis=1).reshape(-1)
        return y_pred
    def adjust_weights(self, x):
        """
        调整layer的权重
        """
        dist_mat = np.zeros((len(x), len(self.layers)))
        for layer_id, layer in enumerate(self.layers):
            layer_center = layer.layer_center
            dist = get_dist(x[:, layer.layer_gene_idx], layer_center)
            dist = np.reshape(dist, -1)
            dist_mat[:, layer_id] = dist
        weight_mat =get_weights(dist_mat)
        return weight_mat

    def bypass_predict(self, x_origin, x, layer_id):
        new_hard_idx = []
        new_easy_idx = []
        return new_hard_idx, new_easy_idx

    def bypass_train(self, y_all, y_layer_pred, sample_idx):
        """
        sample_idx: 传入本layer的样本索引
        """
        threshold = 0.2
        y_layer_true = y_all[sample_idx]
        error = self._error(y_layer_true, y_layer_pred)
        easy_sample_idx = sample_idx[np.argwhere(error <= threshold).ravel()]
        hard_sample_idx = sample_idx[np.argwhere(error > threshold).ravel()]

        if self.is_keep_exsample is True:
            # self._hold_extra_sample(y_all, hard_sample_idx, easy_sample_idx)
            new_hard_idx, new_easy_idx = self._hold_extra_sample(y_all[sample_idx], hard_sample_idx, easy_sample_idx)
            # easy_sample_idx, hard_sample_idx = self._hold_extra_sample(y_all, hard_sample_idx, easy_sample_idx)
        return hard_sample_idx, easy_sample_idx

    def dropout(self, term_scores_dict, threshold):
        """dropout, 为每一个term剔除性能差的孩子节点, 更新term_child_map之后, 再更新self.child_layer_list, self.gene_layer_list
        parameters
        ----------
            term_scores_dict: 孩子节点对应的得分
            threshold: 保留term的百分比
        return
        ------
            None
        """
        pruned_term_child_map = self._pruned_term_child(self.term_child_map_used, term_scores_dict, threshold=threshold)
        return pruned_term_child_map

    def record_term_child_map_info(self):
        """记录term_child_map的信息
        
        return
        ------
            info_df: DataFrame
        """
        assert hasattr(self, "term_child_map_used"), "the object has no attr 'term_child_map'"
        info_list = []
        index = []
        for layer_id, layer in enumerate(self.term_layer_list):
            for term in layer:
                index.append(term)
                child_list = self.term_child_map_used.get(term)
                # info = (layer, len(child_list), child_list)
                info_list.append((layer_id, len(child_list), child_list))         
        info_df = pd.DataFrame(info_list, index=index, columns=['layer_id', 'children number', 'child_list'])
        return info_df

    def _pair_gene(self, term_gene_dict: dict):
        term_gene_paired_dict = {}
        for term, gene in term_gene_dict.items():
            gene_pair = list(gene) + list(np.array(gene) + self.gene_dim)
            term_gene_paired_dict[term] = tuple(gene_pair)
        return term_gene_paired_dict

    def _get_term_child_map(self):
        """
        获取每个term的后辈节点map
        """
        term_child_map = {}  # term_child_map记录了term的所有后辈节点
        for layer_id, layer in enumerate(self.term_layer_list):
            for term in layer:
                # if term == self.root: 
                #     print(1)
                # if term not in self.term_neighbor_map: continue
                child = deepcopy(self.term_neighbor_map[term])
                for neighbor in self.term_neighbor_map[term]:
                    if neighbor in term_child_map:
                        child.extend(term_child_map[neighbor])
                term_child_map[term] = tuple(np.unique(child))  # tuple防止被篡改

        return term_child_map

    def _init_weighter(self, y_true, term_predict_map: dict):
        """初始化加权器
        
        parameters
        ----------
            y_true: 
            term_predict_map: 
        """
        term_scores_dict = get_term_score_dict_rr(y_true, term_predict_map, 'r2')
        term_names, term_val_scores = [], []
        for term_name, score in term_scores_dict.items():
            term_names.append(term_name)
            term_val_scores.append(score)
        weighter = RRWeighter(individual_names=term_names, individual_scores=term_val_scores)
        return weighter

    def _get_neighbor_layer_list(self):
        """
        获取term-neighbor字典的list
        """
        neighbor_layer_list = []
        for layer_list in self.term_layer_list:
            keys = layer_list
            values = itemgetter(*layer_list)(self.term_neighbor_map)
            if len(keys) == 1:  # 对仅含一个元组的列表所作的特殊操作: 转化为[()]的形式
                values = [(values)]
            neighbor_dict = {key: value for key, value in zip(keys, values)}
            neighbor_layer_list.append(neighbor_dict)
        return neighbor_layer_list

    def _get_child_layer_list(self, use_poster):
        """
        获取每一层孩子节点的list
        """
        if use_poster == 'all':
            term_child_map = self._get_term_child_map()
        elif use_poster == 'neighbor':
            term_child_map = self.term_neighbor_map
        else:
            term_child_map = {term_name:[] for term_name in self.term_neighbor_map.keys()}
        # 如果有dropout操作, 则直接覆盖上面
        if hasattr(self, "term_child_map") and self.term_child_map is not None:
            term_child_map = self.term_child_map
        # 记录所使用的term_child_map-Dropout使用
        self.term_child_map_used = term_child_map  # used表示该对象所使用的term_child_map
        save_variable(self.log_dir, self.term_child_map_used, variable_name=["term_child_map_uesd"])
        # 根据所使用的term_child_map, 生成需要的list形式
        layer_child_list = []
        for layer_id in range(self.n_layers):
            keys: list = self.term_layer_list[layer_id]
            l_dict = {key: term_child_map[key] for key in keys}
            layer_child_list.append(l_dict)
        return layer_child_list


    def _get_gene_layer_list_with_poster(self, use_poster='all'):
        """
        use_poster: 是否使用后辈节点, 'neighbor', 'all'(default)
        """
        gene_layer_list = []

        if use_poster == 'all':
            child_dict_list = self._get_child_layer_list(use_poster)
        elif use_poster == 'neighbor':
            child_dict_list = self.neighbor_layer_list
        else:
            child_dict_list = None
        # print(child_dict_list[1]['GO:0007276'])
        if child_dict_list is None:
            gene_layer_list = self._get_gene_layer_list()
        else:
            for layer_id, child_dict in enumerate(child_dict_list):
                gene_dict = {}
                for term, children in child_dict.items():
                    gene_keys = np.append([term], children)
                    # gene_keys.extend([children])
                    gene_values = itemgetter(*gene_keys)(self.term_direct_gene_map)
                    if len(children) != 0:
                        term_gene = np.unique([index for gene_tuple in gene_values for index in gene_tuple])
                    else:
                        term_gene = np.unique(gene_values)
                    gene_dict[term] = term_gene
                gene_layer_list.append(gene_dict)

        return gene_layer_list

    def _get_cpboost_length(self, cpboost_dict: dict):
        """获取每一层cpboost特征的长度
        """
        index = []
        length = []
        for layer_id, feature in cpboost_dict.items():
            # res_dict[layer_id] = np.shape(feature)[1]
            index.append(layer_id)
            length.append(np.shape(feature)[1])
        pd.DataFrame(length, index=index).to_csv(self.log_dir+'/cpboost_length.csv')

    def _get_boost_vec_number(self):
        """拟合完成后, 获取每个节点的增强向量的长度
        """
        with pd.ExcelWriter(self.log_dir+'/boostfeaturenumber.xlsx') as writer:
            for i, layer in enumerate(self.layers):
                index = []
                n_boost = []
                for term_name, term_node in layer.node_dict.items():
                    index.append(term_name)
                    if 'n_boost_features' not in dir(term_node):
                        n_boost.append(0)
                    else:
                        n_boost.append(term_node.n_boost_features)
                    pd.DataFrame(data=n_boost, index=index).to_excel(writer, sheet_name=f'layer_{i}')
            
    def _get_gene_layer_list(self):
        """
        获取每一层的term-gene(直接注释)字典
        """
        # 创建仅含自身term-gene的字典 列表
        gene_layer_list = []
        for term_layer in self.term_layer_list:
            keys = term_layer
            values = list(itemgetter(*keys)(self.term_direct_gene_map))
            if len(keys) == 1:  # 对仅含一个元组的列表所作的特殊操作: 转化为[()]的形式
                values = [(values)]
            l_dict = {key: value for key, value in zip(keys, values)}
            gene_layer_list.append(l_dict)
        return gene_layer_list

    def _hold_extra_sample(self, y_true, hard_idx, easy_idx):
        """
        保留极端样本
        """
        hard_idx = np.copy(hard_idx)
        easy_idx = np.copy(easy_idx)
        if len(y_true) != len(hard_idx) + len(easy_idx):
            return
        extra_idx = np.argwhere(np.abs(y_true) > self.EXTRA_THRESHOLD).ravel()
        easy_idx = np.setdiff1d(easy_idx, extra_idx)
        hard_idx = np.union1d(hard_idx, extra_idx)
        return hard_idx, easy_idx

    def _update_lcb_vector(self, layer_id, lcb_vector_dict, stage):
        """更新layer-level的字典
        parameters
        ---------- 
            layer_id: int
            lcb_vector_dict: dict
                存储layer_compressed_boost_vector的字典
            stage: str, 'train' or 'predict'
        return
            lcb_vector_dict: dict
        """
        if layer_id > 1:    # 只有第2层才开始有层级增强向量(从0开始)
            lcb_vector_dict.update(self.layers[layer_id].make_layer_boost_vec(stage, self.lb_vector_compress_method, self.lb_vector_compress_rate))              # 只需要保存前五层的 压缩增强向量
        return lcb_vector_dict
    
    def explain_sample(self, x, y):
        pass


    @staticmethod
    def _error(y_true, y_pred):
        percentile_error = np.abs((y_true - y_pred) / y_true)
        return percentile_error

    @classmethod
    def _embed_boost_vec_dict(cls, boost_vec_dict: dict, sample_idx, full_idx):
        """
        将得到的层级boost_vec嵌入到全索引的boost_vec中去, 返回全索引的boost_vec
        """
        new_boost_vec_dict = {}
        if cls.check_boost_vec_dim(boost_vec_dict, sample_idx):
            for term, boost_vec in boost_vec_dict.items():
                embed_vec = np.zeros((len(full_idx), np.shape(boost_vec)[1]))
                embed_vec[sample_idx] = boost_vec
                new_boost_vec_dict[term] = embed_vec

        return new_boost_vec_dict

    @classmethod
    def _draw_boost_vec_dict(cls, boost_vec_dict: dict, target_idx, full_idx):
        """
        从全索引的boost_vec取出target_idx的boost_vec
        return
        ------
            new_boost_vec_dict: dict   
                用于训练的增强向量字典, 键: term_name, 值: 索引(hard_idx)对应的样本增强向量
        """
        if boost_vec_dict == {}:
            return boost_vec_dict
        # 检查boost_vec_dict 中的维度是否等于full_idx
        new_boost_vec_dict = {}
        if cls.check_boost_vec_dim(boost_vec_dict, full_idx):
            for term, boost_vec in boost_vec_dict.items():
                new_boost_vec_dict[term] = boost_vec[target_idx]
        else: 
            print("boost_vec_dict中的维度不等于full_idx")

        return new_boost_vec_dict

    @staticmethod
    def check_boost_vec_dim(boost_vec_dict, index):
        # check(boost_vec, sample_idx)
        for rand_term_boost_vec in boost_vec_dict.values():
            if len(rand_term_boost_vec) != len(index):
                print()
                assert False, "增强向量和sample_idx维度不一致! "
            else:
                return True

    @staticmethod
    def predict_weighted(y_mat, weight_mat):
        weight_mat.reshape((-1, 1))
        y_pred = np.sum(y_mat*weight_mat, axis=0)
        return y_pred

    @staticmethod
    def _pruned_term_child(term_child_map, term_scores_dict, threshold=None):
        """针对各个term单独进行剪枝
        parameters
        ----------
            term_child_map: dict
                孩子节点的列表
            term_scores_dict: dict
                孩子节点对应的得分
            threshold: float, 0 < threshold < 1
                保留孩子节点的比例
        return
        ------
            pruned_term_child_map: 修剪之后的term_child_map
        """
        pruned_term_child_map = {}
        for term, child_list in term_child_map.items():
            if len(child_list) == 0: 
                pruned_term_child_map[term] = []
                continue
            # 先获取child_list对应的score_list
            score_list = []
            for child in child_list:
                score_list.append(term_scores_dict[child])
            # 将得分低于均值的child剔除
            if threshold is None:
                thshd_score = np.mean(score_list)
            else:
                if len(child_list)>=2: 
                    score_list_sorted = np.sort(score_list)[::-1]
                    temp = score_list_sorted[:np.ceil(len(child_list)*threshold).astype(int)]
                    thshd_score = temp[-1]
                else:
                    thshd_score = score_list[0]
            pruned_child_list = np.array(child_list)[np.argwhere(np.array(score_list)>=thshd_score).reshape(-1)]
            pruned_child_list = pruned_child_list.tolist()
            pruned_term_child_map[term] = pruned_child_list
        root_child = pruned_term_child_map["GO:0008150"]
        print(f"dropout后根节点孩子数量: {len(root_child)}")
        return pruned_term_child_map

class GOForestClassifier:
    EXTRA_THRESHOLD: int = 15
    CUDA_ID: int = 3
    RECON_ERROR_THRESHOLD: float = 0.25

    def __init__(
            self,
            term_layer_list: list,
            term_direct_gene_map: dict,  # 假设value是list, 而不是ndarray
            term_neighbour_map: dict,
            gene_dim: int,  # gene维度, 955
            root: str,
            forest_configs: dict,
            layer_configs: dict,
            node_configs=None,
            **kwargs,
    ):
        """
        kwargs: 
            term_child_map
        """
        if kwargs:
            self.kwargs = kwargs
            self.term_child_map = self.kwargs.get("term_child_map")
        
        self.log_dir = node_configs['log_dir']
        self.root = root
        self.gene_dim = gene_dim
        self.n_layers = len(term_layer_list)
        self.term_layer_list = term_layer_list
        self.term_neighbor_map = term_neighbour_map
        
        self.easy_threshold = forest_configs.get("easy_threshold")
        self.hard_thresholds = []
        self.is_bypass_train = forest_configs['is_bypass_train']
        self.is_keep_exsample = forest_configs['is_keep_exsample']
        self.is_bypass_predict = forest_configs['is_bypass_predict']
        self.is_weighting_predict = forest_configs['is_weighting_predict']
        self.poster_gene = forest_configs['poster_gene']      # 'neighbor' or 'all' or 'None'
        self.poster_child = forest_configs['poster_child']      # 'neighbor' or 'all' or 'None'
        self.is_compress_lb_vector = forest_configs["is_compress_lb_vector"]
        self.lb_vector_compress_rate = forest_configs["lb_vector_compress_rate"]
        self.lb_vector_compress_method = forest_configs["lb_vector_compress_method"]
        self.scaler = forest_configs.get("scaler")          # 用于缩放权重的参数, 默认50
        
        self.term_layer_map = self._init_term_layer_map()
        self.term_direct_gene_map = self._pair_gene(term_direct_gene_map)
        self.neighbor_layer_list = self._get_neighbor_layer_list()
        self.child_layer_list = self._get_child_layer_list(self.poster_child)
        self.gene_layer_list = self._get_gene_layer_list_with_poster(self.poster_gene)
        self.layers:List[LayerCLF] = self._init_layer(layer_configs, node_configs)

        self.logger = self._init_logger(os.path.join(self.log_dir, 'log'))
    
    def _init_logger(self, filename):
        """初始化日志器"""
        import logging
        logger = logging.getLogger('mylogger')
        logger.setLevel("DEBUG")
        fhandle = logging.FileHandler(filename)
        fhandle.setLevel("INFO")
        formater = logging.Formatter(fmt="%(asctime)s - %(name)s - %(message)s")
        fhandle.setFormatter(formater)
        logger.addHandler(fhandle)

        streamhandle = logging.StreamHandler()
        streamhandle.setLevel('INFO')
        streamhandle.setFormatter(formater)
        logger.addHandler(fhandle)

        return logger

    def _init_term_layer_map(self):
        """初始化term_layer映射的字典, 键值对为term_name: layer_id
        """
        term_layer_map = {}
        for layer_id, layer in enumerate(self.term_layer_list):
            for term_name in layer:
                term_layer_map[term_name] = layer_id
        return term_layer_map

    def _get_node(self, term_name: str):
        """根据term_name获取node对象
        
        parameters
        ----------
            term_name: str, term's name
            
        return
        ------
            node: Node, term_name对应的Node对象
        """
        layer_id = self.term_layer_map[term_name]
        return self.layers[layer_id].node_dict[term_name]

    def _init_layer(self, layer_configs, node_configs):
        layer_list = []
        for layer_id in range(self.n_layers):
            l = LayerCLF(
                layer_id,
                self.term_layer_list[layer_id],
                self.gene_layer_list[layer_id],
                self.child_layer_list[layer_id],
                layer_configs,
                node_configs
            )
            layer_list.append(l)
        return layer_list
    
    def _build_x_y_train(self, x_origin, y_origin, train_idx, x_art, y_art):
        """
        """
        if x_art != []:
            return np.vstack([x_origin[train_idx], x_art]), np.hstack([y_origin[train_idx], y_art])
        else:
            return x_origin[train_idx], y_origin[train_idx]

    def fit(self, x, y, phychem=None, fp=None):
        y = np.squeeze(y)
        self.n_class = len(np.unique(y))
        x_origin = deepcopy(x)
        y_origin = deepcopy(y)
        full_idx_origin = np.arange(len(x_origin))
        hard_idx = np.arange(len(x_origin))
        hard_idx_list = []
        easy_idx_list = []
        error_idx_list = []
        train_idx_list = []     # 记录每一层有哪些原始样本被训练
        nob_dict_origin = {}               # 训练阶段的增强向量字典
        proba_dict = {}       # 训练阶段的类概率字典
        lcb_dict_origin = {}     # 训练阶段的cpboost_layer的字典
        x_art, y_art = [], []   # 利用困难数据人造的新样本
        self.confidence_thresholds = []
        y_pred_layer_mean_list = []
        train_idx_origin = np.arange(len(x_origin))   # 训练使用的原始样本的索引
        x_train, y_train = deepcopy(x_origin), deepcopy(y_origin)
        y_proba = np.zeros((y_origin.shape[0], self.n_class))
        for layer_id, layer in enumerate(self.layers):
            train_idx_list.append(train_idx_origin)
            x_train, y_train = self._build_x_y_train(x_origin, y_origin, train_idx_origin, x_art, y_art)
            nob_in, lcb_in = self._build_boost_vec_train(nob_dict_origin, lcb_dict_origin, train_idx_origin, layer_id, x_art)
            if phychem is not None: 
                phychem = phychem[hard_idx]
            if fp is not None:
                fp = fp[hard_idx]
            if layer_id < self.n_layers-1:
                proba_pred_layer_mean, proba_layer_mat, nob_out_layer, proba_layer_dict = layer.fit(
                    x_train, y_train,
                    nob_in=nob_in,
                    lcb_in=lcb_in,
                )
                if self.is_compress_lb_vector:
                    lcb_dict_origin = self._update_lcb_dict(layer_id, lcb_dict_origin, 'train', train_idx_origin, full_idx_origin)
            else:
                proba_pred_layer_mean, proba_layer_mat, nob_out_layer, proba_layer_dict = layer.fit(
                    x_train, y_train,
                    nob_in=nob_in,
                    lcb_in=lcb_in,
                    phychem=phychem, fp=fp
                )
            y_pred_layer_mean = np.argmax(proba_pred_layer_mean, axis=1)
            y_pred_layer_mean_list.append(y_pred_layer_mean)
            nob_out_layer: dict = self._embed_nob_dict(
                nob_out_layer, train_idx_origin, full_idx_origin
            )
            nob_dict_origin.update(nob_out_layer)
            proba_layer_dict = self._embed_nob_dict(
                proba_layer_dict, train_idx_origin, full_idx_origin
            )
            proba_dict.update(proba_layer_dict)
            y_proba[train_idx_origin] = proba_pred_layer_mean[:len(train_idx_origin)]

            if self.is_bypass_train:
                proba_mat_origin = np.empty((x_origin.shape[0], proba_layer_mat.shape[1]))
                proba_mat_origin[train_idx_origin] = proba_layer_mat[:len(train_idx_origin)]
                hard_idx, easy_idx, y_pred_mean = self.divide_hard_easy(
                # hard_idx, easy_idx, y_pred_mean = self.divide_hard_easy2(
                    y_origin, proba_mat_origin, train_idx_origin, layer_id
                )

                if x_art == []:
                    x_art = deepcopy(x_origin[hard_idx])
                    y_art = deepcopy(y_origin[hard_idx])
                else:
                    x_art = np.vstack([x_art, x_origin[hard_idx]])
                    y_art = np.hstack([y_art, y_origin[hard_idx]])
                easy_idx_list.append(easy_idx)
                hard_idx_list.append(hard_idx)

                # train_idx_origin 不应该增加, 而是应该减少, 删除一些简单样本
                train_idx_origin = np.setdiff1d(train_idx_origin, easy_idx)

                pd.Series(hard_idx).to_csv(os.path.join(self.log_dir, f"训练集_hard_idx_layer{layer_id}.csv"))
            else:
                easy_idx_list.append([])
                hard_idx_list.append(train_idx_origin)
            print(f"训练完第{layer_id}层")
            # 输出精度
            from sklearn.metrics import f1_score, roc_auc_score
            y_true = y_origin[train_idx_origin]
            y_pred = np.argmax(proba_pred_layer_mean, axis=1)[train_idx_origin]
            self.logger.info(f"训练数据上: 第{layer_id}层平均预测的roc_auc: {roc_auc_score(y_true, y_pred):.3f}\tf1: {f1_score(y_true, y_pred):.3f}")
            y_pred = np.argmax(y_proba, axis=1)
            self.logger.info(f"所有原始数据上: 第{layer_id}层平均预测的roc_auc: {roc_auc_score(y_origin, y_pred):.3f}\tf1: {f1_score(y_origin, y_pred):.3f}")
            # 记录错分类样本的序号
            error_idx_list.append(train_idx_origin[y_true!=y_pred].squeeze())
        
        # 把难分样本, 错分类样本收集, 训练一个纠错器
        unique = lambda x: np.unique(np.hstack(x))
        idx_rec = unique([unique(train_idx_list), unique(error_idx_list)])
        from xgboost import XGBClassifier
        rec_model = XGBClassifier().fit(x_origin[idx_rec], y_origin[idx_rec])
        self.rec_model: XGBClassifier = rec_model

        # 加权预测
        if self.is_weighting_predict:
            # 先将类别概率转化成预测值
            term_predict_map = {}
            y_pred = np.empty_like(y_origin)
            for layer_id, (train_idx, easy_idx) in enumerate(zip(train_idx_list, easy_idx_list)):
                for term in self.term_layer_list[layer_id]:
                    # proba = proba_dict[term][0:len(train_idx)]
                    proba = proba_dict[term][train_idx]
                    y_pred[train_idx] = np.argmax(proba, axis=1)                # 替换训练样本的数据
                    y_true = y_origin
                    term_predict_map[term] = np.transpose([y_true, y_pred])
                y_pred[easy_idx] = y_pred_layer_mean_list[layer_id][easy_idx]             # 在进入下层的计算之前, 把本层的易分样本的结果固定
            # 初始化加权器
            self.weighter = self._init_weighter(y_origin, term_predict_map)
        
        # 保存训练中间结果
        save_variable(self.log_dir, nob_dict_origin, variable_name=['nob_dict_train'])
        save_variable(self.log_dir, lcb_dict_origin, variable_name=['lcb_dict_train'])
        self._get_boost_vec_number()
        self._get_cpboost_length(lcb_dict_origin)
        self._get_est_params()
        
        return nob_dict_origin, easy_idx_list, hard_idx_list

    def predict(self, x, y=None, phychem=None, fp=None):
        proba, proba_mat, final_layer_proba = self.predict_proba(x, y, phychem=phychem, fp=fp)
        y_pred = np.argmax(proba, axis=1)
        final_layer_predict = np.argmax(final_layer_proba, axis=1)
        y_pred_mat = np.transpose([np.argmax(item, axis=1) for item in proba_mat])

        save_variable(self.log_dir, y_pred, final_layer_predict, y_pred_mat, variable_name=['y_pred', 'final_layer_predict', 'y_pred_mat'])
        return y_pred

    def predict_proba(self, x, y=None, phychem=None, fp=None):
        """
        
        Return
        ------
        y_proba: shape(n_samples, n_classes)
        proba_mat: shape(n_terms, n_samples, n_classes)
        final_layer_proba: shape(n_samples, n_classes)
        """
        x_origin = deepcopy(x)
        full_idx = np.arange(len(x_origin))
        hard_idx = np.arange(len(x_origin))
        easy_idx = []
        hard_idx_list = []
        easy_idx_list = []
        predict_idx_list = []
        nob_dict = {}         # 预测阶段的 增强向量字典
        lcb_dict = {}     # 预测阶段的 cpboost_layer 字典
        y_proba_dict = {}           # 预测阶段的 proba 字典
        y_proba = np.zeros((len(x), self.n_class))
        predict_idx = np.arange(len(x_origin))
        # proba_mat = np.empty((x_origin.shape[0], 0))
        for layer_id, layer in enumerate(self.layers):
            predict_idx_list.append(predict_idx)
            x = x_origin[predict_idx]
            nob_in, lcb_in = self._build_boost_vec_predict(nob_dict, lcb_dict, predict_idx, full_idx)
            nob_out_layer = {}
            if layer_id < self.n_layers-1:
                layer_pred_proba, proba_layer_mat, nob_out_layer, y_proba_layer_dict = layer.predict_proba(
                    x, nob_in, lcb_in
                )
                if self.is_compress_lb_vector:
                    lcb_dict = self._update_lcb_dict(layer_id, lcb_dict, 'predict', predict_idx, full_idx)
            else:
                layer_pred_proba, proba_layer_mat, nob_out_layer, y_proba_layer_dict = layer.predict_proba(
                    x, nob_in, lcb_in, phychem, fp
                )
            y_proba[predict_idx] = layer_pred_proba
            nob_out_layer = self._embed_nob_dict(nob_out_layer, predict_idx, full_idx)
            nob_dict.update(nob_out_layer)
            y_proba_layer_dict = self._embed_nob_dict(y_proba_layer_dict, predict_idx, full_idx)
            y_proba_dict.update(y_proba_layer_dict)

            if self.is_bypass_predict:
                proba_mat = np.empty((x_origin.shape[0], proba_layer_mat.shape[1]))
                # proba_mat = 
                proba_mat[predict_idx] = proba_layer_mat
                hard_idx, easy_idx = self.divide_hard_easy_predict(
                # hard_idx, easy_idx = self.divide_hard_easy_predict2(
                    proba_mat, predict_idx, layer_id, y
                )
                hard_idx_list.append(hard_idx)
                easy_idx_list.append(easy_idx)
                # 保留易分样本的结果, 不再进入下层预测
                predict_idx = np.setdiff1d(predict_idx, easy_idx)
            else:
                hard_idx_list.append(predict_idx)
                easy_idx_list.append([])
            
            # 输出精度
            if y is not None:
                from sklearn.metrics import f1_score, roc_auc_score
                y_pred = np.argmax(y_proba, axis=1)
                self.logger.info(f"全体测试集上: 第{layer_id}层平均预测的roc_auc: {roc_auc_score(y, y_pred):.3f}\tf1: {f1_score(y, y_pred):.3f}")
        
        # 记录最后一层的预测结果
        final_layer_proba = deepcopy(y_proba)
        
        # 加权预测, 仅针对最后一层的predict_idx的样本做加权预测
        if self.is_weighting_predict:
            scaler = self.scaler if self.scaler is not None else 50
            weighted_proba = self.weighter.weight_predict_proba(y_proba_dict, scaler=scaler)
            y_proba[predict_idx] = weighted_proba[predict_idx]
            save_variable(self.log_dir, self.weighter.weights, y_proba, variable_name=['nodes_weight', 'weighted_pred_proba'])
            self.weighter.weights.to_csv(os.path.join(self.log_dir, "结点weights.csv"))
        else:
            y_proba = np.mean([value for value in y_proba_dict.values()], axis=0)
        # 将y_proba归一化
        y_proba /= np.sum(y_proba, axis=1).reshape((-1,1))
        
        # 调节预测结果
        # y_proba = self._confidence_adjust(y_proba, nob_dict)

        # 保存中间预测结果
        proba_mat = []
        for value in y_proba_dict.values():
            proba_mat.append(value)
        self.pred_proba_mat = proba_mat # shape=(n_term, n_sample, n_class)
        save_variable(self.log_dir, nob_dict, variable_name=['boost_vec_dict_predict_stage'])
        
        # 保存增强向量的字典
        self.nob_dict = nob_dict
        self.lcb_dict = lcb_dict

        if self.is_bypass_predict and y is not None:
        # 使用rec学习器调整难分样本的预测结果
            idx = np.unique(np.hstack(hard_idx_list))
            y_true = y[idx]
            rec_proba = self.rec_model.predict_proba(x_origin[idx])
            rec_pred = np.argmax(rec_proba, axis=1)
            self.logger.info(f"纠错学习器在难分样本上的roc_auc: {roc_auc_score(y_true, rec_pred):.3f}\tf1: {f1_score(y_true, rec_pred):.3f}")
            final_layer_pred = np.argmax(final_layer_proba[idx], axis=1)
            self.logger.info(f"最后一层学习器在难分样本上的roc_auc: {roc_auc_score(y_true, final_layer_pred):.3f}\tf1: {f1_score(y_true, final_layer_pred):.3f}")
            weighted_pred = np.argmax(y_proba[idx], axis=1)
            self.logger.info(f"加权器在难分样本上的roc_auc: {roc_auc_score(y_true, weighted_pred):.3f}\tf1: {f1_score(y_true, weighted_pred):.3f}")
            mean_pred = self.predict_average()[idx]
            self.logger.info(f"平均预测在难分样本上的roc_auc: {roc_auc_score(y_true, mean_pred):.3f}\tf1: {f1_score(y_true, mean_pred):.3f}")
        return y_proba, proba_mat, final_layer_proba
  
    def predict_average(self):
        """
        返回各层的平均预测值
        """
        if self.pred_proba_mat is not None:
            mean_pred_proba = np.mean(self.pred_proba_mat, axis=0)
            y_pred_average = np.argmax(mean_pred_proba, axis=1)
            save_variable(self.log_dir, y_pred_average, variable_name=['y_pred_mean_layer'])
            return y_pred_average.reshape(-1)

    def predict_average_proba(self):
        """
        返回各层的平均预测概率
        
        Returns
        -------
        mean_pred_proba: ndarray, shape=(n_sample, n_class). 
            通过平均所有结点的类概率值得到的类概率值.
        proba_voting: ndarray, shape=(n_sample, n_class). 
            通过统计投票结果而转化的类概率值.
        """
        if self.pred_proba_mat is not None:
            # 平均proba
            mean_pred_proba = np.mean(self.pred_proba_mat, axis=0)
            # 统计投票结果, 然后转化为proba
            voting_mat = [np.argmax(proba, axis=1) for proba in self.pred_proba_mat]
            proba_class1 = np.sum(voting_mat, axis=0)/len(voting_mat)
            proba_class0 = 1 - proba_class1
            proba_voting = np.vstack([proba_class0, proba_class1]).T
            return mean_pred_proba, proba_voting
            
    def _confidence_adjust(self, y_pred_proba_weighted, y_pred_proba_node:dict):
        """置信度调节预测值

        Parameters
        ----------
        y_pred_proba_weighted: ndarray, shape=(#sample, #class)
        y_pred_proba_node: dict, nob_dict of data

        Returns
        -------
        y_pred_proba_new: ndarray, shape=(#sample, #class)
        """
        def get_confidence(y_pred_proba):
            """获取置信度函数, 返回预测结果的置信度, shape=(# sample,)
            """
            label = np.argmax(y_pred_proba, axis=1)
            confidence = 2*np.abs(y_pred_proba[np.arange(y_pred_proba.shape[0]),label]) - 1
            return confidence

        # 获得节点预测概率矩阵
        columns = []
        proba_array = []
        for term, probas in y_pred_proba_node.items():
            columns.extend([f"{term}_class_{label}" for label in [0, 1]])
            proba_array.append(probas)
        proba_df = pd.DataFrame(np.hstack(proba_array), columns=columns)

        # 获取加权预测的低置信度样本索引(模型预测)
        confidence_model_pred = get_confidence(y_pred_proba_weighted)
        low_confidence_idx = np.argwhere(confidence_model_pred<0.6).reshape(-1)

        # 获得样本的置信度矩阵, shape = (# sample, # term), (结点预测的置信度)
        pred_low_confidence = proba_df.iloc[low_confidence_idx, :].values  # 低置信度预测矩阵
        confidence_matrix = []
        for i in range(int(pred_low_confidence.shape[1]/2)):
            confidence_matrix.append(get_confidence(pred_low_confidence[:, i*2:(i+1)*2]))
        confidence_matrix = np.transpose(confidence_matrix)

        # 把置信度矩阵中小于0.5的结点预测屏蔽
        idx = np.argwhere(confidence_matrix<0.6)
        row_idx, col_idx = idx[:,0], idx[:, 1]
        confidence_matrix[row_idx, col_idx] = 0 

        # 置信度乘上原始权重
        # confidence_matrix = confidence_matrix*weights*1092

        # 获取结点对样本的预测矩阵
        y_pred_matrix = []
        for i in range(int(pred_low_confidence.shape[1]/2)):
            proba = pred_low_confidence[:, i*2:(i+1)*2]
            y_pred_matrix.append(np.argmax(proba, axis=1))
        y_pred_matrix = np.transpose(y_pred_matrix)

        # 计算新的预测结果
        y_proba_low_confidence_new = []
        for label in (0,1):
            mat = np.zeros_like(y_pred_matrix, dtype=np.float64)
            idx = np.argwhere(y_pred_matrix==label)
            row_idx, col_idx = idx[:,0], idx[:, 1]
            mat[row_idx, col_idx] = confidence_matrix[row_idx, col_idx]

            y_proba_low_confidence_new.append(np.sum(mat, axis=1))
        y_proba_low_confidence_new = np.transpose(y_proba_low_confidence_new)
        y_proba_low_confidence_new /= np.sum(y_proba_low_confidence_new, axis=1).reshape((-1,1))

        # 更新预测结果
        y_pred_proba_new = deepcopy(y_pred_proba_weighted)
        y_pred_proba_new[low_confidence_idx] = y_proba_low_confidence_new

        return y_pred_proba_new

    def _pair_gene(self, term_gene_dict: dict):
        term_gene_paired_dict = {}
        for term, gene in term_gene_dict.items():
            gene_pair = list(gene) + list(np.array(gene) + self.gene_dim)
            term_gene_paired_dict[term] = tuple(gene_pair)
        return term_gene_paired_dict

    def _get_term_child_map(self):
        """
        获取每个term的后辈节点map
        """
        term_child_map = {}  # term_child_map记录了term的所有后辈节点
        for layer_id, layer in enumerate(self.term_layer_list):
            for term in layer:
                # if term == self.root: 
                #     print(1)
                # if term not in self.term_neighbor_map: continue
                child = deepcopy(self.term_neighbor_map[term])
                for neighbor in self.term_neighbor_map[term]:
                    if neighbor in term_child_map:
                        child.extend(term_child_map[neighbor])
                term_child_map[term] = tuple(np.unique(child))  # tuple防止被篡改

        return term_child_map

    def _get_gene_layer_list(self):
        """
        获取每一层的term-gene(直接注释)字典
        """
        # 创建仅含自身term-gene的字典 列表
        gene_layer_list = []
        for term_layer in self.term_layer_list:
            keys = term_layer
            values = list(itemgetter(*keys)(self.term_direct_gene_map))
            if len(keys) == 1:  # 对仅含一个元组的列表所作的特殊操作: 转化为[()]的形式
                values = [(values)]
            l_dict = {key: value for key, value in zip(keys, values)}
            gene_layer_list.append(l_dict)
        return gene_layer_list

    def _get_neighbor_layer_list(self):
        """
        获取term-neighbor字典的list
        """
        neighbor_layer_list = []
        for layer_list in self.term_layer_list:
            keys = layer_list
            values = itemgetter(*layer_list)(self.term_neighbor_map)
            if len(keys) == 1:  # 对仅含一个元组的列表所作的特殊操作: 转化为[()]的形式
                values = [(values)]
            neighbor_dict = {key: value for key, value in zip(keys, values)}
            neighbor_layer_list.append(neighbor_dict)
        return neighbor_layer_list

    def _get_child_layer_list(self, use_poster):
        """
        获取每一层孩子节点的list
        """
        if use_poster == 'all':
            term_child_map = self._get_term_child_map()
        elif use_poster == 'neighbor':
            term_child_map = self.term_neighbor_map
        else:
            term_child_map = {term_name:[] for term_name in self.term_neighbor_map.keys()}
        # 如果有dropout操作, 则直接覆盖上面
        if hasattr(self, "term_child_map") and self.term_child_map is not None:
            term_child_map = self.term_child_map
        # 记录所使用的term_child_map-Dropout使用
        self.term_child_map_used = term_child_map  # used表示该对象所使用的term_child_map
        save_variable(self.log_dir, self.term_child_map_used, variable_name=["term_child_map_uesd"])
        # 根据所使用的term_child_map, 生成需要的list形式
        layer_child_list = []
        for layer_id in range(self.n_layers):
            keys: list = self.term_layer_list[layer_id]
            l_dict = {key: term_child_map[key] for key in keys}
            layer_child_list.append(l_dict)
        return layer_child_list

    def _get_gene_layer_list_with_poster(self, use_poster='all'):
        """
        use_poster: 是否使用后辈节点, 'neighbor', 'all'(default)
        """
        gene_layer_list = []

        if use_poster == 'all':
            child_dict_list = self._get_child_layer_list(use_poster)
        elif use_poster == 'neighbor':
            child_dict_list = self.neighbor_layer_list
        else:
            child_dict_list = None
        # print(child_dict_list[1]['GO:0007276'])
        if child_dict_list is None:
            gene_layer_list = self._get_gene_layer_list()
        else:
            for layer_id, child_dict in enumerate(child_dict_list):
                gene_dict = {}
                for term, children in child_dict.items():
                    gene_keys = np.append([term], children)
                    # gene_keys.extend([children])
                    gene_values = itemgetter(*gene_keys)(self.term_direct_gene_map)
                    if len(children) != 0:
                        term_gene = np.unique([index for gene_tuple in gene_values for index in gene_tuple])
                    else:
                        term_gene = np.unique(gene_values)
                    gene_dict[term] = term_gene
                gene_layer_list.append(gene_dict)
        return gene_layer_list
        
    def _build_boost_vec_train(self, nob_dict:dict, lcb_dict: dict, origin_idx, layer_id, art_samples):
        """构建输入增强特征"""
        def _build_boost_vec_in_origin(nob_dict:dict, lcb_dict: dict, origin_idx):
            """为原始的样本构建nob和lcb字典"""
            nob_in = {}
            lcb_in = {}
            # 构建nob
            for term_name, nob in nob_dict.items():
                nob_in[term_name] = nob[origin_idx]
            # 构建lcb
            for layer_id, lcb in lcb_dict.items():
                lcb_in[layer_id] = lcb[origin_idx]
            return nob_in, lcb_in

        def _build_boost_vec_in_art(layer_id, art_samples):
            """为新的样本构建nob和lcb字典"""
            nob_in = {}
            lcb_in = {}
            for i in range(layer_id):
                layer = self.layers[i]
                _, _, nob_out_layer, _ = layer.predict_proba(art_samples, nob_in, lcb_in)
                if self.is_compress_lb_vector:
                    self._update_lcb_dict(i, lcb_in, 'predict')
                nob_in.update(nob_out_layer)
            return nob_in, lcb_in
        
        nob_in_origin, lcb_in_origin = _build_boost_vec_in_origin(nob_dict, lcb_dict, origin_idx)
        
        if self.is_bypass_train:
            nob_in, lcb_in = {}, {}
            nob_in_art, lcb_in_art = _build_boost_vec_in_art(layer_id, art_samples)
            for term_name in nob_in_origin.keys():
                nob_in[term_name] = np.vstack([nob_in_origin[term_name], nob_in_art[term_name]])
            for layer_id in lcb_in_origin.keys():
                lcb_in[layer_id] = np.vstack([lcb_in_origin[layer_id], lcb_in_art[layer_id]])
            return nob_in, lcb_in
        else:
            return nob_in_origin, lcb_in_origin

    def _build_boost_vec_predict(self, nob_dict, lcb_dict, predict_idx, full_idx):
        """构建预测的增强向量"""
        def _build_boost_vec_in_origin(nob_dict:dict, lcb_dict: dict, origin_idx):
            """为原始的样本构建nob和lcb字典"""
            nob_in = {}
            lcb_in = {}
            # 构建nob
            for term_name, nob in nob_dict.items():
                nob_in[term_name] = nob[origin_idx]
            # 构建lcb
            for layer_id, lcb in lcb_dict.items():
                lcb_in[layer_id] = lcb[origin_idx]
            return nob_in, lcb_in
        nob_in, lcb_in = _build_boost_vec_in_origin(nob_dict, lcb_dict, predict_idx)
        return nob_in, lcb_in

    def _update_lcb_dict(self, layer_id, lcb_dict, stage, sample_idx=None, full_idx=None):
        """更新layer-level的字典
        parameters
        ---------- 
            layer_id: int
            lcb_vector_dict: dict
                存储layer_compressed_boost_vector的字典
            stage: str, 'train' or 'predict'
        return
            lcb_vector_dict: dict
        """
        if layer_id > 1:    # 只有第2层才开始有层级增强向量(从0开始)
            lcb_out_dict = self.layers[layer_id].make_layer_boost_vec(stage, self.lb_vector_compress_method, self.lb_vector_compress_rate)          # 只需要保存前五层的 压缩增强向量
            if sample_idx is not None:
                for layer_id, lcb in lcb_out_dict.items():
                    lcb_out_dict[layer_id] = np.zeros((len(full_idx), lcb.shape[1]))
                    lcb_out_dict[layer_id][sample_idx] = lcb[0:len(sample_idx)]
            lcb_dict.update(lcb_out_dict)              # 只需要保存前五层的 压缩增强向量
        return lcb_dict

    def _init_weighter(self, y_origin, term_predict_map: dict):
        """初始化加权器
        
        Parameters
        ----------
        y_true: y的真实值
        term_predict_map: dict, term预测的字典, 
            键是term_name, 值是ndarray, shape=(#sample, 2), 存储y_true和y_pred
        """
        def get_term_score_dict_clf(y_true, predict_dict: dict, metric_name='f1_score'):
            from sklearn.metrics import f1_score
            metric = eval(metric_name)
            term_score_dict = {}
            for term, y_true_pred in predict_dict.items():
                y_true = y_true_pred[:,0]
                y_pred = y_true_pred[:,1]
                term_score_dict[term] =  metric(y_true, y_pred)
            return term_score_dict
        
        term_scores_dict = get_term_score_dict_clf(y_origin, term_predict_map, 'f1_score')
        term_names, term_val_scores = [], []
        for term_name, score in term_scores_dict.items():
            term_names.append(term_name)
            term_val_scores.append(score)
        weighter = CLFWeighter(individual_names=term_names, individual_scores=term_val_scores)
        return weighter

    def divide_hard_easy(self, y_origin, proba_mat: np.ndarray, sample_idx, layer_id=None):
        """
        仅仅针对原始样本划分难分易分样本, 不包括重复的样本和人造的新样本.
        重复的样本虽然在基因特征上相同, 但是在增强特征上会略有差异
        Paramters
        ---------
        y_origin
        proba_mat: ndarray, the probability of all the nodes in a layer, 
            shape=(n_sample, n_node)
        sample_idx: 原始样本的真索引

        Returns
        -------
        hard_sample_idx: ndarray, 样本在原数据中的真索引
        easy_sample_idx: ndarray
        """
        
        from sklearn.metrics import accuracy_score
        def get_confidence(y_pred_proba):
            """计算预测概率的置信度"""
            label = np.argmax(y_pred_proba, axis=1)
            confidence = 2*np.abs(y_pred_proba[np.arange(y_pred_proba.shape[0]),label])-1
            return confidence
        def count_rows(arr, condition):
            return np.apply_along_axis(lambda x: np.count_nonzero(condition(x)), 1, arr)
        
        proba_mat = proba_mat[sample_idx]
        y_true = y_origin[sample_idx]
        n_sample, n_node = len(sample_idx), int(proba_mat.shape[1]/self.n_class)
        # 计算置信度矩阵; 
        # 计算置信度的阈值, 划分预测为高置信度还是低置信度的实数, 在(0,1)之间
        # 阈值的两种方案: 1-每个结点平均预测的accuracy; 2-正确预测的样本的平均概率
        y_pred_mat, confidence_mat = np.empty((n_sample, n_node)), np.empty((n_sample, n_node))
        confidence_threshold = 0
        for i in range(n_node):
            pred_proba = proba_mat[:,i*self.n_class:(i+1)*self.n_class]
            y_pred_mean = np.argmax(pred_proba, axis=1)
            y_pred_mat[:, i] = y_pred_mean
            confidence_threshold += accuracy_score(y_true, y_pred_mean)
            confidence_mat[:,i] = get_confidence(pred_proba)
        confidence_threshold /= n_node
        confidence_threshold *= 0.8
        self.confidence_thresholds.append(confidence_threshold)
        # 统计每个样本的预测有多少是高可信, 有多少是低可信
        low_confidence_count = count_rows(
            confidence_mat, 
            lambda x: x<confidence_threshold,
        )
        # 计算每个样本的难分度=low_count/n_node, 是衡量这个样本是否难分的标准, 
        hard_rate = low_confidence_count/n_node
        hard_threshold = confidence_threshold
        easy_threshold = 0.02 if self.easy_threshold is None else self.easy_threshold
        # 划分出难分样本和易分样本的索引
        hard_sample_idx = np.atleast_1d(
            sample_idx[np.argwhere(hard_rate>hard_threshold).squeeze()]
        )
        easy_sample_idx = np.atleast_1d(
            sample_idx[np.argwhere(hard_rate<=easy_threshold).squeeze()]
        )
        # normalize = lambda x: x/np.sum(x)
        if np.size(easy_sample_idx) != 0:
            easy_sample_idx = np.random.choice(
                easy_sample_idx, 
                len(hard_sample_idx),
                # p = normalize(1/normalize(np.where(temp==0, 2*np.min(temp), temp)))
            )

        # 将中间结果保存下来 列名:[样本序号, 真实值, 平均投票的预测结果, 样本置信度, 是否难分]
        y_proba_mean = 0
        for i in range(n_node):
            y_proba_mean += proba_mat[:,i*self.n_class:(i+1)*self.n_class]
        y_pred_mean = np.argmax(y_proba_mean, axis=1)
        cache = pd.DataFrame(np.transpose([y_true, y_pred_mean, hard_rate, np.zeros_like(y_pred_mean)]), columns=['y_true', 'y_pred', '难分度', '是否难分'], index=sample_idx)
        cache.loc[hard_sample_idx, "是否难分"] = 1
        cache.loc[easy_sample_idx, "是否易分"] = 1
        cache["平均投票的预测结果是否正确"] = y_true==y_pred_mean
        cache["置信度阈值"] = confidence_threshold
        cache.to_csv(os.path.join(self.log_dir, f"难分样本情况_layer{layer_id}.csv"))
        
        return np.unique(hard_sample_idx), np.unique(easy_sample_idx), y_pred_mean

    def divide_hard_easy2(self, y_origin, proba_mat: np.ndarray, sample_idx, layer_id=None):
        """
        仅仅针对原始样本划分难分易分样本, 不包括重复的样本和人造的新样本.
        重复的样本虽然在基因特征上相同, 但是在增强特征上会略有差异
        Paramters
        ---------
        y_origin
        proba_mat: ndarray, the probability of all the nodes in a layer, 
            shape=(n_sample, n_node)
        sample_idx: 原始样本的真索引

        Returns
        -------
        hard_sample_idx: ndarray, 样本在原数据中的真索引
        easy_sample_idx: ndarray
        """
        
        from sklearn.metrics import accuracy_score
        def get_confidence(y_pred_proba):
            """计算预测概率的置信度--直接使用概率值当置信度"""
            confidence = np.max(y_pred_proba, axis=1)
            return confidence
        def count_rows(arr, condition):
            return np.apply_along_axis(lambda x: np.count_nonzero(condition(x)), 1, arr)
        
        proba_mat = proba_mat[sample_idx]
        y_true = y_origin[sample_idx]
        n_sample, n_node = len(sample_idx), int(proba_mat.shape[1]/self.n_class)
        # 计算置信度矩阵; 
        # 计算置信度的阈值, 划分预测为高置信度还是低置信度的实数, 在(0,1)之间
        # 置信度阈值的两种方案: 1-每个结点平均预测的accuracy; 2-正确预测的样本的平均概率
        hard_threshold = 0
        y_pred_mat, confidence_mat = np.empty((n_sample, n_node)), np.empty((n_sample, n_node))
        for i in range(n_node):
            pred_proba = proba_mat[:,i*self.n_class:(i+1)*self.n_class]
            y_pred_node = np.argmax(pred_proba, axis=1)
            y_pred_mat[:, i] = y_pred_node
            hard_threshold += accuracy_score(y_true, y_pred_node)
            confidence_mat[:,i] = get_confidence(pred_proba)
        confidence_threshold = np.mean(confidence_mat)
        self.confidence_thresholds.append(confidence_threshold)
        # 统计每个样本的预测有多少是高可信, 有多少是低可信
        low_confidence_count = count_rows(
            confidence_mat, 
            lambda x: x<confidence_threshold,
        )
        # 计算每个样本的难分度hard_rate=low_count/n_node, 是衡量这个样本是否难分的标准, 
        hard_rate = low_confidence_count/n_node
        hard_threshold /= n_node
        self.hard_thresholds.append(hard_threshold)
        easy_threshold = 0.1 if self.easy_threshold is None else self.easy_threshold

        # 划分出难分样本和易分样本的索引
        hard_sample_idx = np.atleast_1d(
            sample_idx[np.argwhere(hard_rate>hard_threshold).squeeze()]
        )
        easy_sample_idx = np.atleast_1d(
            sample_idx[np.argwhere(hard_rate<=easy_threshold).squeeze()]
        )
        # 对错分样本, 也应该加入难分样本的索引
        y_proba_mean = 0
        for i in range(n_node):
            y_proba_mean += proba_mat[:,i*self.n_class:(i+1)*self.n_class]
        y_pred_mean = np.argmax(y_proba_mean, axis=1)
        error_sample_idx = np.atleast_1d(
            sample_idx[y_true!=y_pred_mean]
        )
        hard_sample_idx = np.union1d(hard_sample_idx, error_sample_idx)
        easy_sample_idx = np.setdiff1d(easy_sample_idx, error_sample_idx)

        # 随机抽取出难分样本数量的易分样本剔除
        # normalize = lambda x: x/np.sum(x)
        if np.size(easy_sample_idx) != 0:
            easy_sample_idx = np.random.choice(
                easy_sample_idx, 
                len(hard_sample_idx),
                # p = normalize(1/normalize(np.where(temp==0, 2*np.min(temp), temp)))
            )

        # 将中间结果保存下来 列名:[样本序号, 真实值, 平均投票的预测结果, 样本置信度, 是否难分]
        # y_proba_mean = 0
        # for i in range(n_node):
        #     y_proba_mean += proba_mat[:,i*self.n_class:(i+1)*self.n_class]
        # y_pred_mean = np.argmax(y_proba_mean, axis=1)
        cache = pd.DataFrame(np.transpose([y_true, y_pred_mean, hard_rate, np.zeros_like(y_pred_mean)]), columns=['y_true', 'y_pred', '难分度', '是否难分'], index=sample_idx)
        cache.loc[hard_sample_idx, "是否难分"] = 1
        cache.loc[easy_sample_idx, "是否易分"] = 1
        cache["平均投票的预测结果是否正确"] = y_true==y_pred_mean
        cache["置信度阈值"] = confidence_threshold
        cache["难分样本阈值"] = hard_threshold
        cache.to_csv(os.path.join(self.log_dir, f"难分样本情况_layer{layer_id}.csv"))
        
        return np.unique(hard_sample_idx), np.unique(easy_sample_idx), y_pred_mean

    def divide_hard_easy_predict(self, proba_mat: np.ndarray, sample_idx, layer_id, y_test):
        def get_confidence(y_pred_proba):
            """计算预测概率的置信度"""
            label = np.argmax(y_pred_proba, axis=1)
            confidence = 2*np.abs(y_pred_proba[np.arange(y_pred_proba.shape[0]),label])-1
            return confidence
        def count_rows(arr, condition):
            return np.apply_along_axis(lambda x: np.count_nonzero(condition(x)), 1, arr)

        proba_mat = proba_mat[sample_idx]
        n_sample, n_node = len(sample_idx), int(proba_mat.shape[1]/self.n_class)
        # 计算置信度矩阵; 
        y_pred_mat, confidence_mat = np.empty((n_sample, n_node)), np.empty((n_sample, n_node))
        confidence_threshold = self.confidence_thresholds[layer_id]
        for i in range(n_node):
            pred_proba = proba_mat[:,i*self.n_class:(i+1)*self.n_class]
            y_pred_mean = np.argmax(pred_proba, axis=1)
            y_pred_mat[:, i] = y_pred_mean
            confidence_mat[:,i] = get_confidence(pred_proba)

        # 统计每个样本的预测有多少是高可信, 有多少是低可信
        low_confidence_count = count_rows(
            confidence_mat, 
            lambda x: x<confidence_threshold,
        )
        # 计算每个样本的难分度=low_count/n_node, 是衡量这个样本是否难分的标准, 
        hard_rate = low_confidence_count/n_node
        hard_threshold = confidence_threshold
        easy_threshold = 0.02
        # 划分出难分样本和易分样本的索引
        hard_sample_idx = np.atleast_1d(
            sample_idx[np.argwhere(hard_rate>hard_threshold).squeeze()]
        )
        easy_sample_idx = np.atleast_1d(
            sample_idx[np.argwhere(hard_rate<=easy_threshold).squeeze()]
        )
        # normalize = lambda x: x/np.sum(x)
        if np.size(easy_sample_idx) != 0:
            easy_sample_idx = np.random.choice(
                easy_sample_idx, 
                len(hard_sample_idx),
                # p = normalize(1/normalize(hard_rate[easy_sample_idx]))
            )

        # 将中间结果保存下来 列名:[样本序号, 真实值, 平均投票的预测结果, 样本置信度, 是否难分]
        cache = pd.DataFrame(np.transpose([y_pred_mean, hard_rate]), columns=['y_pred', '难分度'], index=sample_idx)
        cache.loc[hard_sample_idx, "是否难分"] = 1
        cache.loc[easy_sample_idx, "是否易分"] = 1
        cache.loc[sample_idx, "是否预测正确"] = y_pred_mean==y_test[sample_idx]
        cache["置信度阈值"] = confidence_threshold
        cache.to_csv(os.path.join(self.log_dir, f"预测集_难分样本情况_layer{layer_id}.csv"))

        return hard_sample_idx, easy_sample_idx    

    def divide_hard_easy_predict2(self, proba_mat: np.ndarray, sample_idx, layer_id, y_test):
        def get_confidence(y_pred_proba):
            """计算预测概率的置信度--直接使用概率值当置信度"""
            confidence = np.max(y_pred_proba, axis=1)
            return confidence
        def count_rows(arr, condition):
            return np.apply_along_axis(lambda x: np.count_nonzero(condition(x)), 1, arr)

        proba_mat = proba_mat[sample_idx]
        n_sample, n_node = len(sample_idx), int(proba_mat.shape[1]/self.n_class)
        # 计算置信度矩阵; 
        y_pred_mat, confidence_mat = np.empty((n_sample, n_node)), np.empty((n_sample, n_node))
        confidence_threshold = self.confidence_thresholds[layer_id]
        for i in range(n_node):
            pred_proba = proba_mat[:,i*self.n_class:(i+1)*self.n_class]
            y_pred_mean = np.argmax(pred_proba, axis=1)
            y_pred_mat[:, i] = y_pred_mean
            confidence_mat[:,i] = get_confidence(pred_proba)

        # 统计每个样本的预测有多少是高可信, 有多少是低可信
        low_confidence_count = count_rows(
            confidence_mat, 
            lambda x: x<confidence_threshold,
        )
        # 计算每个样本的难分度=low_count/n_node, 是衡量这个样本是否难分的标准, 
        hard_rate = low_confidence_count/n_node
        hard_threshold = self.hard_thresholds[layer_id]
        easy_threshold = 0.1
        # 划分出难分样本和易分样本的索引
        hard_sample_idx = np.atleast_1d(
            sample_idx[np.argwhere(hard_rate>hard_threshold).squeeze()]
        )
        easy_sample_idx = np.atleast_1d(
            sample_idx[np.argwhere(hard_rate<=easy_threshold).squeeze()]
        )
        # normalize = lambda x: x/np.sum(x)
        if np.size(easy_sample_idx) != 0:
            easy_sample_idx = np.random.choice(
                easy_sample_idx, 
                len(hard_sample_idx),
                # p = normalize(1/normalize(hard_rate[easy_sample_idx]))
            )

        # 将中间结果保存下来 列名:[样本序号, 真实值, 平均投票的预测结果, 样本置信度, 是否难分]
        cache = pd.DataFrame(np.transpose([y_pred_mean, hard_rate]), columns=['y_pred', '难分度'], index=sample_idx)
        cache.loc[hard_sample_idx, "是否难分"] = 1
        cache.loc[easy_sample_idx, "是否易分"] = 1
        cache.loc[sample_idx, "是否预测正确"] = y_pred_mean==y_test[sample_idx]
        cache["置信度阈值"] = confidence_threshold
        cache.to_csv(os.path.join(self.log_dir, f"预测集_难分样本情况_layer{layer_id}.csv"))

        return hard_sample_idx, easy_sample_idx    

    def _get_cpboost_length(self, cpboost_dict: dict):
        """获取每一层cpboost特征的长度
        """
        index = []
        length = []
        for layer_id, feature in cpboost_dict.items():
            # res_dict[layer_id] = np.shape(feature)[1]
            index.append(layer_id)
            length.append(np.shape(feature)[1])
        pd.DataFrame(length, index=index).to_csv(self.log_dir+'/lcb的长度.csv')

    def _get_boost_vec_number(self):
        """拟合完成后, 获取每个节点的增强向量的长度
        """
        with pd.ExcelWriter(self.log_dir+'/nob的长度.xlsx') as writer:
            for i, layer in enumerate(self.layers):
                index = []
                n_boost = []
                for term_name, term_node in layer.node_dict.items():
                    index.append(term_name)
                    if 'n_boost_features' not in dir(term_node):
                        n_boost.append(0)
                    else:
                        n_boost.append(term_node.n_boost_features)
                    pd.DataFrame(data=n_boost, index=index).to_excel(writer, sheet_name=f'layer_{i}')

    def _get_proba_dict(self, proba_mat):
        proba_dict = {}
        i = 0
        for layer in self.term_layer_list:
            for term in layer:
                proba_dict[term] = proba_mat

    def _get_est_params(self):
        """获取node中基学习器的参数"""
        params_list = []
        for term_name, layer_id in self.term_layer_map.items():
            node = self._get_node(term_name)
            params_se = pd.Series(node.est_configs)
            params = params_se.values
            params = np.append([term_name, layer_id], params_se.values)
            params_list.append(params)
        columns = np.append(["term's name", "layer id"], params_se.index)
        
        pd.DataFrame(params_list, columns=columns).to_csv(os.path.join(self.log_dir, "每个结点的xgbs参数.csv"))

    def record_term_child_map_info(self):
        """记录term_child_map的信息
        
        return
        ------
            info_df: DataFrame
        """
        assert hasattr(self, "term_child_map_used"), "the object has no attr 'term_child_map'"
        info_list = []
        index = []
        for layer_id, layer in enumerate(self.term_layer_list):
            for term in layer:
                index.append(term)
                child_list = self.term_child_map_used.get(term)
                # info = (layer, len(child_list), child_list)
                info_list.append((layer_id, len(child_list), child_list))         
        info_df = pd.DataFrame(info_list, index=index, columns=['layer_id', 'children number', 'child_list'])
        return info_df

    def dropout(self, term_scores_dict, threshold):
        """dropout, 为每一个term剔除性能差的孩子节点, 更新term_child_map之后, 再更新self.child_layer_list, self.gene_layer_list
        parameters
        ----------
            term_scores_dict: 孩子节点对应的得分
            threshold: 保留term的百分比
        return
        ------
            None
        """
        pruned_term_child_map = self._pruned_term_child(self.term_child_map_used, term_scores_dict, threshold=threshold)
        return pruned_term_child_map


    @classmethod
    def _embed_nob_dict(cls, boost_vec_dict: dict, sample_idx, full_idx):
        """
        将得到的层级boost_vec嵌入到全索引的boost_vec中去, 返回全索引的boost_vec
        """
        sample_idx = np.unique(sample_idx)    # train_idx一定是full_idx的子集, 但是可能会存在某些索引出现多次(困难样本的索引)
        new_boost_vec_dict = {}
        for term, boost_vec in boost_vec_dict.items():
            empty_boost_vec = np.zeros((len(full_idx), np.shape(boost_vec)[1]))
            empty_boost_vec[sample_idx] = boost_vec[:len(sample_idx)]
            new_boost_vec_dict[term] = empty_boost_vec
        return new_boost_vec_dict
    
    @staticmethod
    def _pruned_term_child(term_child_map, term_scores_dict, threshold=None):
        """针对各个term单独进行剪枝
        parameters
        ----------
            term_child_map: dict
                孩子节点的列表
            term_scores_dict: dict
                孩子节点对应的得分
            threshold: float, 0 < threshold < 1
                保留孩子节点的比例
        return
        ------
            pruned_term_child_map: 修剪之后的term_child_map
        """
        pruned_term_child_map = {}
        for term, child_list in term_child_map.items():
            if len(child_list) == 0: 
                pruned_term_child_map[term] = []
                continue
            # 先获取child_list对应的score_list
            score_list = []
            for child in child_list:
                score_list.append(term_scores_dict[child])
            # 将得分低于均值的child剔除
            if threshold is None:
                thshd_score = np.mean(score_list)
            else:
                if len(child_list)>=2: 
                    score_list_sorted = np.sort(score_list)[::-1]
                    temp = score_list_sorted[:np.ceil(len(child_list)*threshold).astype(int)]
                    thshd_score = temp[-1]
                else:
                    thshd_score = score_list[0]
            pruned_child_list = np.array(child_list)[np.argwhere(np.array(score_list)>=thshd_score).reshape(-1)]
            pruned_child_list = pruned_child_list.tolist()
            pruned_term_child_map[term] = pruned_child_list
        root_child = pruned_term_child_map["GO:0008150"]
        print(f"dropout后根节点孩子数量: {len(root_child)}")
        return pruned_term_child_map



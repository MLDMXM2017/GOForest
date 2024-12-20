import numpy as np

from sklearn import pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from .evaluation import scores_rr
from .node_add_rotation import NodeRR, NodeCLF
from .util import get_dist, get_weights, save_variable
from .weighter import CLFWeighter
from typing import List

class LayerRR:
    TRAIN_STAGE: str = 'train'
    PREDICT_STAGE: str = 'predict'
    def __init__(
        self,
        layer_id,
        term_list,
        term_gene_map,
        term_child_map,
        layer_configs: dict,
        node_configs: dict,
    ):
        self.log_dir = node_configs['log_dir']
        self.layer_id = layer_id
        self.term_list = term_list
        self.term_gene_map = term_gene_map
        self.layer_center = None
        self.adjust_weighting_predict: bool = layer_configs['is_weighting_predict']  # 调整node的权重, 加权预测
        # self.compress_rate: float = layer_configs.get('compress_rate') if layer_configs.get('compress_rate') else 0.9 # 
        self.layer_gene_idx = np.unique([index for item in self.term_gene_map.values() for index in item])
        self.node_dict = self._init_nodes(term_gene_map, term_child_map, node_configs)

    def _init_nodes(self, term_gene_map, term_child_map, node_configs):
        node_dict = {}
        for term in self.term_list:
            node_dict[term] = NodeRR(
                term,
                self.layer_id,
                feature_idx=term_gene_map[term],
                child_list=term_child_map[term],
                node_configs=node_configs
            )
        return node_dict

    def fit(self, x, y, boost_vec_dict_in, layer_cpboost_dict, phychem=None, fp=None):
        """
        return:
        boost_vec_dict_layer: dict of boost_vec of this layer in train stage
        """
        boost_vec_layer = {}
        y_pred_val_mat = [] # shape: (node_num, sample_num)
        layer_cpboost_array = self._process_cpboost_dict(layer_cpboost_dict)
        for term in self.term_list:
            boost_vec_term_out = self.node_dict[term].fit(x, y, boost_vec_dict_in, layer_cpboost_array, phychem, fp)
            boost_vec_layer[term] = boost_vec_term_out
            y_pred_val_mat.append(np.mean(boost_vec_term_out, axis=1))
        y_pred_val_layer_mean = np.mean(y_pred_val_mat, 0)
        self.layer_center = self._get_centers(x, True, True)

        return y_pred_val_layer_mean, boost_vec_layer

    def predict(self, x, boost_vec_dict, layer_cpboost_dict, phychem=None, fp=None):
        """
        return:
        y_pred: 1-dim
        boost_vec_dict_layer: dict of boost_vec of this layer in predict stage
        """
        boost_vec_layer = {}
        y_pred_mat = []
        layer_cpboost_array = self._process_cpboost_dict(layer_cpboost_dict)
        for node in self.node_dict.values():
            term_name = node.term_name
            node_pred, node_boost_vec = node.predict(x, boost_vec_dict, layer_cpboost_array, phychem, fp)
            # node_pred, node_boost_vec = node.predict_multi_process(x, boost_vec_dict, layer_cpboost_array, phychem, fp)
            boost_vec_layer[term_name] = node_boost_vec
            y_pred_mat.append(node_pred)
        y_pred_mat = np.transpose(y_pred_mat)   # mat shape: (sample number, node number)
        save_variable(self.log_dir, y_pred_mat, variable_name=[f'nodes_predict_layer_{self.layer_id}'])     # 保存这一层所有node的预测结果
        if self.adjust_weighting_predict:
            # x_gene = x[:, self.layer_gene_idx]
            weight_mat = self.adjust_weights(x)
            y_pred = np.sum(y_pred_mat * weight_mat, axis=1)
        else:
            y_pred = np.mean(y_pred_mat, axis=1)
        return y_pred, boost_vec_layer

    def _get_centers(self, x_train, layer_center: bool = True, sub_feature: bool = True) -> np.ndarray:
        """
        get the center vector of this layer
        """
        if layer_center:
            if sub_feature:
                gene_idx = self.layer_gene_idx
                layer_center = np.mean(x_train[:, gene_idx], axis=0)
            else:
                layer_center = np.mean(x_train, axis=0)
        return layer_center

    def adjust_weights(self, x_test):
        """
        调整 节点的权重
        """
        # dist_mat = get_dist(x_test, self.layer_center)
        node_dist_mat = np.zeros((len(x_test), len(self.term_list)))   # shape: (sample num, node num)
        for node_seq, node in enumerate(self.node_dict.values()):
            node_center = node.node_center
            dist = get_dist(x_test, node_center)
            dist = np.reshape(dist, -1)
            node_dist_mat[:, node_seq] = dist

        weight_mat = get_weights(node_dist_mat)
        return weight_mat

    def get_scores(self, x, y, boost_vec_dict, phychem, fp):
        y_pred, _ = self.predict(x, boost_vec_dict, phychem, fp)
        score_df = scores_rr(y, y_pred)
        return score_df

    def _process_cpboost_dict(self, cpboost_dict: dict):
        """将cpboost的字典转化成cpboost array
        return 
        ------
            cpboost_array: array-like (n_sample, n_cpboost)
        """
        cpboost_array = np.array([])
        for layer_id, cpboost in cpboost_dict.items():
            if len(cpboost_array) == 0:
                cpboost_array = cpboost
            else:
                cpboost_array = np.hstack((cpboost_array, cpboost))
        return cpboost_array

    def make_layer_boost_vec(self, stage, method='mean', compress_rate=0.85):
        """
        获取本层的压缩增强向量

        parameters
        ----------
            stage: str, 'train' or 'predict'
            method: str, 'mean' or 'linear_convert', default is 
            compress_rate: float, default is 0.85
        return
        ------
            cpboost_dict: dict
                键: layer_id
                值: ndarray like (n_sample, n_cpboost of this layer)
        """
        assert (stage == self.TRAIN_STAGE) or (stage == self.PREDICT_STAGE), "stage 参数必须是'TRAIN'或'PREDICT'"
        ncb = []
        # 训练阶段
        if stage == self.TRAIN_STAGE:
            _attr = 'ncb_train'
            for term_name, term_node in self.node_dict.items():
                if _attr in dir(term_node):
                    ncb = term_node.ncb_train if len(ncb) == 0 else np.hstack((ncb, term_node.ncb_train)) 
            assert np.shape(ncb)[0] != 0, ""
            if np.shape(ncb)[1] > 1:  # 
                if method == "linear_convert":
                    self.boost_compressor = pipeline.make_pipeline(
                            StandardScaler(),
                            PCA(n_components=compress_rate, svd_solver='full')
                        )
                    self.lcb_train = self.boost_compressor.fit_transform(ncb)
                elif method == "mean":
                    self.lcb_train = np.mean(ncb, axis=1).reshape((-1,1))
            elif np.shape(ncb)[1] == 1:
                self.lcb_train = ncb
            cpboost_dict = {self.layer_id : self.lcb_train}
        # 测试阶段
        elif stage == self.PREDICT_STAGE:
            _attr = 'ncb_test'
            for term_name, term_node in self.node_dict.items():
                if _attr in dir(term_node):
                    ncb = term_node.ncb_test if len(ncb) == 0 else np.hstack((ncb, term_node.ncb_test))
            assert np.shape(ncb)[0] != 0
            if np.shape(ncb)[1] > 1:
                if method == "linear_convert":
                    self.lcb_test = self.boost_compressor.transform(ncb)
                elif method == "mean":
                    self.lcb_test = np.mean(ncb, axis=1).reshape((-1, 1))
            elif np.shape(ncb)[1] == 1:
                self.lcb_test = ncb
            cpboost_dict = {self.layer_id : self.lcb_test}

        return cpboost_dict
 

class LayerCLF:
    TRAIN_STAGE: str = 'train'
    PREDICT_STAGE: str = 'predict'
    def __init__(
        self,
        layer_id,
        term_list,
        term_gene_map,
        term_child_map,
        layer_configs: dict,
        node_configs: dict,
    ):
        self.n_class = None
        self.log_dir = node_configs['log_dir']
        self.layer_id = layer_id
        self.term_list = term_list
        self.term_gene_map = term_gene_map
        self.layer_center = None
        self.adjust_weighting_predict: bool = layer_configs['is_weighting_predict']  # 调整node的权重, 加权预测
        # self.compress_rate: float = layer_configs.get('compress_rate') if layer_configs.get('compress_rate') else 0.9 # 
        self.layer_gene_idx = np.unique([index for item in self.term_gene_map.values() for index in item])
        self.node_dict: List[LayerCLF] = self._init_nodes(term_gene_map, term_child_map, node_configs)


    def _init_nodes(self, term_gene_map, term_child_map, node_configs):
        node_dict = {}
        for term in self.term_list:
            node_dict[term] = NodeCLF(
                term,
                self.layer_id,
                feature_idx=term_gene_map[term],
                child_list=term_child_map[term],
                node_configs=node_configs
            )
        return node_dict

    def fit(self, x, y, nob_in: dict, lcb_in: dict, phychem=None, fp=None):
        """
        Parameters
        ----------
        x: features
        y: labels
        nob_in: dict, node origin boost features, term_name:nob_array
        lcb_in: dict, layer compressed boost features, layer_id:lcb_array

        Returns
        -------
        layer_pred_proba_mean:
        boost_vec_layer:
        proba_dict:
        """
        self.n_class = len(np.unique(y))
        nob_out_layer = {}
        y_proba_layer_dict = {}
        y_proba_layer_mat = [] # shape: (#node, #sample, #class)
        layer_cpboost_array = self._process_cpboost_dict(lcb_in)
        for term in self.term_list:
            nob_out_node, y_pred_proba_node = self.node_dict[term].fit(x, y, nob_in, layer_cpboost_array, phychem, fp)
            nob_out_layer[term] = nob_out_node
            y_proba_layer_dict[term] = y_pred_proba_node
            y_proba_layer_mat.append(nob_out_node)
        y_proba_layer_mean = np.mean(y_proba_layer_mat, 0)    # shape(#sample, #class)
        y_proba_layer_mat = np.hstack(y_proba_layer_mat) # shape: (#sample, #node*#class)
        
        self.layer_center = self._get_centers(x, True, True)

        return y_proba_layer_mean, y_proba_layer_mat, nob_out_layer, y_proba_layer_dict

    def predict(self, x, boost_vec_dict, layer_cpboost_dict, phychem=None, fp=None):
        """
        return:
            y_pred: 1-dim
            boost_vec_layer_dict: dict of boost_vec of this layer in predict stage
        """
        y_proba_layer_mean, boost_vec_layer_dict, y_proba_layer_dict = self.predict_proba(self, x, boost_vec_dict, layer_cpboost_dict, phychem, fp)
        y_pred = np.argmax(y_proba_layer_mean, axis=1).reshape((-1))
        
        return y_pred, boost_vec_layer_dict, y_proba_layer_dict

    def predict_proba(self, x, nob_in, lcb_in, phychem=None, fp=None):
        """
        Returns:
        y_proba_layer_mean: layer的平均或者加权预测概率
        boost_vec_layer: dict of boost_vec of this layer in predict stage
        """
        nob_out_layer = {}
        y_proba_layer_dict = {}
        y_proba_layer_mat = []
        layer_cpboost_array = self._process_cpboost_dict(lcb_in)
        for term in self.term_list:
            node = self.node_dict[term]
            y_pred_proba_node, nob_out_node = node.predict_proba(x, nob_in, layer_cpboost_array, phychem, fp)
            nob_out_layer[term] = nob_out_node
            y_proba_layer_dict[term] = y_pred_proba_node
            y_proba_layer_mat.append(nob_out_node)
        y_proba_layer_mean = np.mean(y_proba_layer_mat, 0)
        y_proba_layer_mat = np.hstack(y_proba_layer_mat)
        # y_pred_mat = []
        # save_variable(self.log_dir, y_pred_mat, variable_name=[f'nodes_predict_layer_{self.layer_id}'])     # 保存这一层所有node的预测结果
        # if self.adjust_weighting_predict:
        #     # x_gene = x[:, self.layer_gene_idx]
        #     weight_mat = self.adjust_weights(x)
        #     y_pred = np.sum(y_pred_mat * weight_mat, axis=1)
        # else:
        #     y_pred = np.mean(y_pred_mat, axis=1)
        return y_proba_layer_mean, y_proba_layer_mat, nob_out_layer, y_proba_layer_dict

    def _get_centers(self, x_train, layer_center: bool = True, sub_feature: bool = True) -> np.ndarray:
        """
        get the center vector of this layer
        """
        if layer_center:
            if sub_feature:
                gene_idx = self.layer_gene_idx
                layer_center = np.mean(x_train[:, gene_idx], axis=0)
            else:
                layer_center = np.mean(x_train, axis=0)
        return layer_center

    def adjust_weights(self, x_test):
        """
        调整 节点的权重
        """
        # dist_mat = get_dist(x_test, self.layer_center)
        node_dist_mat = np.zeros((len(x_test), len(self.term_list)))   # shape: (sample num, node num)
        for node_seq, node in enumerate(self.node_dict.values()):
            node_center = node.node_center
            dist = get_dist(x_test, node_center)
            dist = np.reshape(dist, -1)
            node_dist_mat[:, node_seq] = dist

        weight_mat = get_weights(node_dist_mat)
        return weight_mat

    def get_scores(self, x, y, boost_vec_dict, phychem, fp):
        y_pred, _ = self.predict(x, boost_vec_dict, phychem, fp)
        score_df = scores_rr(y, y_pred)
        return score_df

    def _process_cpboost_dict(self, cpboost_dict: dict):
        """将cpboost的字典转化成cpboost array
        
        Returns 
        ------
        cpboost_array: array-like (n_sample, n_cpboost)
        """
        cpboost_array = np.array([])
        for layer_id, cpboost in cpboost_dict.items():
            if len(cpboost_array) == 0:
                cpboost_array = cpboost
            else:
                cpboost_array = np.hstack((cpboost_array, cpboost))
        return cpboost_array

    def make_layer_boost_vec(self, stage, method='mean', compress_rate=0.85):
        """
        获取本层的压缩增强向量

        parameters
        ----------
            stage: str, 'train' or 'predict'
            method: str, 'mean' or 'linear_convert', default is 
            compress_rate: float, default is 0.85
        return
        ------
            cpboost_dict: dict
                键: layer_id
                值: ndarray like (n_sample, n_cpboost of this layer)
        """
        assert (stage == self.TRAIN_STAGE) or (stage == self.PREDICT_STAGE), "stage 参数必须是'TRAIN'或'PREDICT'"
        ncb = []
        # 训练阶段
        if stage == self.TRAIN_STAGE:
            _attr = 'ncb_train'
            for term_name, term_node in self.node_dict.items():
                if _attr in dir(term_node):
                    ncb = term_node.ncb_train if len(ncb) == 0 else np.hstack((ncb, term_node.ncb_train)) 
            assert np.shape(ncb)[0] != 0, ""
            if np.shape(ncb)[1] > 1:  # 如果压缩增强特征的数量大于1
                if method == "linear_convert":
                    self.boost_compressor = pipeline.make_pipeline(
                            StandardScaler(),
                            PCA(n_components=compress_rate, svd_solver='full')
                        )
                    self.lcb_train = self.boost_compressor.fit_transform(ncb)
                elif method == "mean":
                    even_index = np.arange(0, np.shape(ncb)[1], 2, dtype=int)   # 0类概率索引
                    odd_index = np.arange(1, np.shape(ncb)[1], 2, dtype=int)    # 1类概率索引
                    mean_proba0 = np.mean(ncb[:, even_index], axis=1).reshape((-1, 1))
                    mean_proba1 = np.mean(ncb[:, odd_index], axis=1).reshape((-1, 1))
                    self.lcb_train = np.hstack((mean_proba0, mean_proba1))
            elif np.shape(ncb)[1] == 1:
                self.lcb_train = ncb
            cpboost_dict = {self.layer_id : self.lcb_train}
        # 测试阶段
        elif stage == self.PREDICT_STAGE:
            _attr = 'ncb_test'
            for term_name, term_node in self.node_dict.items():
                if _attr in dir(term_node):
                    ncb = term_node.ncb_test if len(ncb) == 0 else np.hstack((ncb, term_node.ncb_test))
            assert np.shape(ncb)[0] != 0
            if np.shape(ncb)[1] > self.n_class:
                if method == "linear_convert":
                    self.lcb_test = self.boost_compressor.transform(ncb)
                elif method == "mean":
                    even_index = np.arange(0, np.shape(ncb)[1], 2, dtype=int)   # 0类概率索引
                    odd_index = np.arange(1, np.shape(ncb)[1], 2, dtype=int)    # 1类概率索引
                    mean_proba0 = np.mean(ncb[:, even_index], axis=1).reshape((-1, 1))
                    mean_proba1 = np.mean(ncb[:, odd_index], axis=1).reshape((-1, 1))
                    self.lcb_test = np.hstack((mean_proba0, mean_proba1))
            elif np.shape(ncb)[1] in (1, self.n_class):
                self.lcb_test = ncb
            cpboost_dict = {self.layer_id : self.lcb_test}

        return cpboost_dict

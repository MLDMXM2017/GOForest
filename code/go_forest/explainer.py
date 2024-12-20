import numpy as np
import pandas as pd
import shap
from copy import deepcopy
from pandas import DataFrame
from .goforest import GOForestRegressor, GOForestClassifier
from .node_add_rotation import NodeRR, NodeCLF
from .node_rule_extractor import NodeRRExtractor, NodeCLFExtractor

class GOFRExplainer():
    def __init__(
        self, 
        GOFR: GOForestRegressor, 
        ):
        self.model: GOForestRegressor = GOFR
        self.log_dir = GOFR.log_dir
        self.root = GOFR.root
        self.gene_dim = GOFR.gene_dim
        self.term_layer_list = GOFR.term_layer_list
        self.n_layers = GOFR.n_layers
        self.term_layer_map = GOFR.term_layer_map
        
        # 获取结点的父亲数量
        term_child_map = self.model.term_child_map_used 
        term_nparent_map = {}
        for children in term_child_map.values():
            for term_name in children:
                if term_nparent_map.get(term_name):
                    term_nparent_map[term_name] += 1
                else:
                    term_nparent_map[term_name] = 1
        self.term_nparent_map = term_nparent_map

    def _get_bc(self, x_gene):
        """bc"""
        self.model.predict(x_gene)
        nob_dict = self.model.nob_dict
        lcb_dict = self.model.lcb_dict
        lcb_array = []
        for key, value in lcb_dict.items():
            lcb_array.append(value)
        lcb_array = np.transpose(np.squeeze(lcb_array))
        return nob_dict, lcb_array

    def _build_node_input(self, term_name, x_gene, nob_dict, lcb_array):
        node: NodeRR = self.model._get_node(term_name)
        x = x_gene[:, node.gene_idx]
        lcb_array = lcb_array[:, :node.layer_id-2] if node.layer_id >= 2 else []
        node_input = node._process_features('predict', x, nob_dict, lcb_array)
        return np.array(node_input)
        
    def _explain_nodes_multi_process(self, term_name, x):
        """解释结点的整体知识, 获取重要性得分, x
        
        Parameters
        ----------
        term_name: str
        x: input for term
        
        Returns
        -------
        importance_df: DataFrame, shape=(number of node's features, 1)
            index: feature names, columns='importance_score'
        """
        from multiprocessing import Process, Manager
        shapley_values_dict = {}
        node: NodeRR = self.model._get_node(term_name)
        feature_names = node.feature_names

        def get_shap(forest_id, res_dict):
            model = node.estimators[forest_id]
            # make explainer
            explainer = shap.explainers.Tree(model, x, feature_names=feature_names)
            # get shap
            shap_value = explainer(x)
            res_dict[str(forest_id)] = shap_value
        
        # 多进程
        manager = Manager()
        res_dict = manager.dict()
        process_list = [Process(target=get_shap, args=(i, res_dict)) for i in range(5)]
        for process in process_list:
            process.start()
        for process in process_list:
            process.join()
        # 处理shapley_values
        shapley_values = [res_dict.get(str(i)) for i in range(5)]
        importances = np.sum([self._calcul_feature_importance(shapley_value) 
            for shapley_value in shapley_values], axis=0)
        importances= importances / np.sum(importances)
        return pd.DataFrame(importances, index=feature_names, columns=['importance_score'])

    def _get_xgb_importance(self, term_name, importance_type='gain'):
        """获取xgb内置的特征重要性
        
        Parameters:
        term_name: str
        importance_type: str, 'gain'(default), 'cover' or 'weight' 
        
        Returns
        -------
        importance_df: DataFrame, shape=(number of node's features, 1)
            index: feature names, columns='importance_score'"""
        node: NodeRR = self.model._get_node(term_name)
        feature_names = node.feature_names
        xgbs = node.estimators
        xgbs = [xgbr.set_params(**{'importance_type':importance_type}) for xgbr in xgbs]
        importances = np.sum([xgbr.feature_importances_ for xgbr in xgbs], axis=0)
        importances = importances / np.sum(importances)
        importance_df = pd.DataFrame(importances, index=feature_names, columns=['importance_score'])

        return importance_df

    def explain_DFS(
        self, 
        x_gene,
        importance_type, 
        importance_method, 
        threshold=0.5
        ): 
        nob_dict, lcb_array = self._get_bc(x_gene)
        term_layer_list = self.term_layer_list
        term_imptc_children_dict = {}      # 记录每个term 对应的重要孩子
        layer_impt_nodes_map = {f"layer{i}":[] for i in range(6)}    # 记录每一层的重要节点
        layer_impt_nodes_map[f"layer5"] = [self.root]
        for i in range(self.n_layers-1, -1, -1):
            for term_name in layer_impt_nodes_map[f"layer{i}"]:
                x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
                if importance_type == 'shap':
                    feature_importance = self._explain_nodes_multi_process(term_name, x)
                else:
                    feature_importance = self._get_xgb_importance(term_name, importance_type)
                    pass

                # 根据importance_df获取重要的nodes
                imptc_nodes_list = self._get_important_nodes_list(
                    feature_importance, 
                    importance_method, 
                    threshold
                )
                # 将term对应的重要children存入字典
                term_imptc_children_dict[term_name] = imptc_nodes_list
                for child in imptc_nodes_list:
                    layer_id = self.term_layer_map[child]
                    layer_impt_nodes_map[f"layer{layer_id}"].append(child)
        
        return layer_impt_nodes_map, term_imptc_children_dict
    
    def get_imptc_path_DFS(self, term_impt_nodes_dict: dict):
        """获取重要路径
        Parameters
        ----------
        term_impt_nodes_dict: dict
            重要节点对应的孩子重要节点"""
        pathways = []
        def recur_search_pathway(cur_term, single_pathway): # DFS
            single_pathway = deepcopy(single_pathway)
            cur_term_layer = self.term_layer_map[cur_term]
            single_pathway.append((cur_term, cur_term_layer))
            if not term_impt_nodes_dict.get(cur_term):
                pathways.append(single_pathway)
                print(single_pathway)
            else:          
                cur_term_children = term_impt_nodes_dict[cur_term]
                for child in cur_term_children:
                    recur_search_pathway(child, single_pathway)

        recur_search_pathway(self.root, [])
        pathways_array = np.zeros((len(pathways), len(self.term_layer_list)), dtype=object)
        for row, pathway in enumerate(pathways):
            for term_name, col in pathway:
                pathways_array[row, col] = term_name
        return pathways_array

    def explain_hierarchy(self, x_gene, importance_type, importance_method, threshold):
        """直接使用每层的节点输出作为特征, 分层找当前层的top重要的节点

        Parameters
        ----------
        importance_type: str, 'shap', 'gain', 'cover' or 'weight'
        importance_method: 0 or 1
            method = 0: 利用数量级作为筛选方案
            method = 1: 对重要性得分排序后, 从最高得分结点开始选择, 逐个加入选择集合, 
            直到集合中的总得分超过设置的阈值
        threshold: float, 0 < threshold < 1

        Returns
        ------
        layer_imptc_list: list[DataFrame], shape=(n_layers, )
            the index of DataFrame element is term name.
            所有层的结点的重要性得分. 
        layer_weight_impotc_list: list[DataFrame], shape=(n_layers, )
        """
        nob_dict, lcb_array = self._get_bc(x_gene)
        from multiprocessing import Process, Manager
        import time
        # initial the dictionary
        imptc_dict = Manager().dict(
            {term : 0.0 for layer in self.term_layer_list for term in layer})
        weighted_imptc_dict = Manager().dict({
            term : 0.0 for layer in self.term_layer_list for term in layer})
        # 获取结点权重
        weights = self.model.weighter.weights * len(self.model.term_direct_gene_map)
        # 获取结点的父亲数量
        term_child_map = self.model.term_child_map_used 
        term_nparent_map = {}
        for children in term_child_map.values():
            for term_name in children:
                if term_nparent_map.get(term_name):
                    term_nparent_map[term_name] += 1
                else:
                    term_nparent_map[term_name] = 1

        # define update function
        def update_dict(nodes_imptc_df: DataFrame):
            for term_name, importance in nodes_imptc_df.iterrows():
                if "GO" in term_name:
                    imptc_dict[term_name] += float(importance) * len(nodes_imptc_df)
        def update_weight_dict(nodes_imptc_df: DataFrame, node_weight: float):
            for term_name, importance in nodes_imptc_df.iterrows():
                if "GO" in term_name:
                    weighted_imptc_dict[term_name] += \
                        float(importance) * node_weight * len(nodes_imptc_df)
                    
        def _sub_process(layer_id, layer):
            """逐层解释的子线程函数"""
            start_time = time.time()
            for term_name in layer:
                x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
                node_weight = float(weights.loc[term_name])
                # 选择解释方案
                if importance_type == 'shap':
                    imptc_df = self._explain_nodes_multi_process(term_name, x)    # 多进程
                else:
                    imptc_df = self._get_xgb_importance(term_name, importance_type)
                    pass
                # 过滤非Gene的重要性
                go_index = imptc_df.index.values[
                    list(map(lambda idx: "GO" in idx, imptc_df.index.values))   
                ]
                nodes_imptc_df = imptc_df.loc[go_index]   # 仅保留GO的特征重要性
                nodes_imptc_df /= nodes_imptc_df.sum()      # 和为1
                update_dict(nodes_imptc_df)
                update_weight_dict(nodes_imptc_df, node_weight)
            end_time = time.time()
            print(f"layer{layer_id} finished, elapsed {end_time-start_time:.2f}s")

        # 多线程解释
        process_list = [
            Process(target=_sub_process, args=(layer_id, layer,)) 
            for layer_id, layer in zip(np.arange(6)[::-1] ,self.term_layer_list[::-1])
        ]
        for process in process_list:
            process.start()
        for process in process_list:
            process.join()

        # 加权的除上每个节点的父亲数量
        for term_name, n_parent in term_nparent_map.items():
            weighted_imptc_dict[term_name] /= n_parent
            imptc_dict[term_name] /= n_parent

        # 将字典分层
        from operator import itemgetter
        layer_imptc_list = []
        layer_weight_imptc_list = []
        for layer_id, layer in enumerate(self.term_layer_list):
            df = pd.DataFrame(
                itemgetter(*layer)(imptc_dict), 
                index=layer, 
                columns=["importance_score"]
            )
            weight_df = pd.DataFrame(
                itemgetter(*layer)(weighted_imptc_dict), 
                index=layer, 
                columns=["importance_score"]
            )
            important_nodes = self._important_metric(
                                    df, 
                                    method=importance_method, 
                                    threshold=threshold
                                    )
            weight_important_nodes = self._important_metric(
                weight_df, 
                method=importance_method,
                threshold=threshold
            )
            df['is important'] = pd.DataFrame(
                [1 if term_name in important_nodes else 0 for term_name in layer], 
                index=layer
            )
            weight_df['is important'] = pd.DataFrame(
                [1 if term_name in weight_important_nodes else 0 for term_name in layer], 
                index=layer
            )
            layer_imptc_list.append(df)
            layer_weight_imptc_list.append(weight_df)
        return layer_imptc_list, layer_weight_imptc_list

    def explain_genes(
        self, 
        x_gene, 
        importance_type, 
        ):
        """获得模型的基因特征重要性

        Parameters
        ----------
        x_gene: array, gene features
        importance_type: str, 'shap', 'gain', 'cover' or 'weight'

        Return
        ------
        importance_df: DataFrame, shape=(n_terms, 1)
            index: term_name
        weighted_importance_df: DataFrame, shape=(n_terms, 1)
            index: term_name
        """
        nob_dict, lcb_array = self._get_bc(x_gene)
        term_direct_gene_map = self.model.term_direct_gene_map
        import time
        from multiprocessing import Process, Manager
        # initial the dictionary
        gene_imptc_dict = Manager().dict(    # 完整的记录结果(不加权)
            {f"Drug{drug_id}_gene{gene_id}":0.0 
                for drug_id in ['A', 'B'] for gene_id in range(955)}) 
        gene_weighted_imptc_dict = Manager().dict(    # 完整的记录结果(不加权)
            {f"Drug{drug_id}_gene{gene_id}":0.0 
                for drug_id in ['A', 'B'] for gene_id in range(955)}) 
        # 获取结点权重
        weights = self.model.weighter.weights * 1092

        # define update function 
        # 调整基因重要性的计算
        def update_dict(gene_imptc_df: DataFrame):
            for index, importance in gene_imptc_df.iterrows():
                if "gene" in index:
                    gene_imptc_dict[index] += float(importance) * len(gene_imptc_df)
        def update_weight_dict(gene_imptc_df: DataFrame, node_weight: float):
            for index, importance in gene_imptc_df.iterrows():
                if "gene" in index:
                    gene_weighted_imptc_dict[index] += \
                        float(importance) * node_weight * len(gene_imptc_df)
                    # print(f"{float(importance)}\t{node_weight}\t{gene_weighted_imptc_dict[index]}")
        def _sub_process(layer_id, layer):
            """逐层解释的子线程函数"""
            start_time = time.time()
            for term_name in layer:
                x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
                node_weight = float(weights.loc[term_name])
                # 选择解释方案
                if importance_type == 'shap':
                    imptc_df = self._explain_nodes_multi_process(term_name, x)    # 多进程
                else:
                    imptc_df = self._get_xgb_importance(term_name, importance_type)
                    pass
                # 过滤非Gene的重要性
                all_gene_index = imptc_df.index.values[
                    list(map(lambda idx: "Drug" in idx, imptc_df.index.values))
                    ]
                direct_gene_index = all_gene_index[
                    list(map(lambda gene_name: int(gene_name.split("gene")[-1]) in 
                        term_direct_gene_map[term_name], all_gene_index))
                    ]
                gene_imptc_df = imptc_df.loc[direct_gene_index]   # 仅保留gene的特征重要性
                gene_imptc_df /= gene_imptc_df.sum()              # 和为1
                update_dict(gene_imptc_df)
                update_weight_dict(gene_imptc_df, node_weight)
            end_time = time.time()
            print(f"layer{layer_id} finished, elapsed {end_time-start_time:.2f}s")

        # 多线程解释Layer
        process_list = [
            Process(target=_sub_process, args=(layer_id, layer,)) 
            for layer_id, layer in zip(np.arange(6)[::-1] ,self.term_layer_list[::-1])
        ]
        for process in process_list:
            process.start()
        for process in process_list:
            process.join()
        
        return gene_imptc_dict, gene_weighted_imptc_dict

    def explain_samples_gene(self, x_gene, importance_type):
        """获取预测数据集中每一个样本的特征重要性

        Parameters
        ----------
        importance_type: str, 'shap', 'gain', 'cover' or 'weight'

        Return
        ------
        importance_df: DataFrame, shape=(n_genes, n_samples)
            index: term_name
        """
        return self.explain_samples_gene_single_process(x_gene, importance_type)

    def explain_samples_gene_single_process(self, x_gene, importance_type):
        """获取预测数据集中每一个样本的特征重要性

        Parameters
        ----------
        importance_type: str, 'shap', 'gain', 'cover' or 'weight'

        Return
        ------
        importance_df: DataFrame, shape=(n_genes, n_samples), 
            index is term_name
        """
        nob_dict, lcb_array = self._get_bc(x_gene)
        import time
        # 初始化结果矩阵: 存储n个样本, 1910个gene特征, 使用Queue在多个子进程之间共享
        n_smaples, n_genes = np.shape(x_gene)
        gene_feature_names = [f"Drug{drug_id}_gene{gene_id}" for drug_id in ['A', 'B'] for gene_id in range(int(n_genes/2))]
        unweight_df = pd.DataFrame(np.zeros((n_smaples, n_genes)), columns=gene_feature_names)
        weight_df = pd.DataFrame(np.zeros((n_smaples, n_genes)), columns=gene_feature_names)
        
        # 获取结点权重
        weights = self.model.weighter.weights * 1092

        for layer_id, layer in zip(np.arange(6)[::-1], self.term_layer_list[::-1]):
            start_time = time.time()
            for term_name in layer:
                x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
                node_weight = float(weights.loc[term_name])
                # 选择解释方案
                if importance_type == 'shap':
                    imptc_df = self._explain_predict_input_single_process(x, term_name)     # 单进程
                else:
                    # assert False, "only 'shap' can passed!"
                    pass
                # 过滤非Gene的重要性
                all_gene_columns = imptc_df.columns.values[
                    list(map(lambda idx: "gene" in idx, imptc_df.columns.values))
                ]
                # 筛选直接注释gene的重要性
                direct_gene_columns = all_gene_columns[list(map(lambda gene_name: int(gene_name.split("gene")[-1]) in self.model.term_direct_gene_map[term_name], all_gene_columns))]
                gene_imptc_df = imptc_df.loc[:, direct_gene_columns]  # 仅保留gene的特征重要性
                gene_imptc_df /= gene_imptc_df.sum(axis=1).to_numpy().reshape((-1,1))   # 和为1

                unweight_df.loc[:, gene_imptc_df.columns] = \
                    gene_imptc_df * gene_imptc_df.shape[1]
                weight_df.loc[:, gene_imptc_df.columns] =\
                     gene_imptc_df * node_weight * gene_imptc_df.shape[1]
            end_time = time.time()
            print(f"layer{layer_id} finished, elapsed {end_time-start_time:.2f}s")

        importance_df = unweight_df
        importance_weight_df = weight_df
        return importance_df.T, importance_weight_df.T

    def explain_samples_term(self, x_gene, importance_type):
        """获取预测数据集中每一个样本决策过程中的增强特征的特征重要性

        Parameters
        ----------
        importance_type: str, 'shap', 'gain', 'cover' or 'weight'

        Returns
        ------
        importance_df: DataFrame, shape=(n_terms, n_samples),
            index: term_name
        """
        return self.explain_samples_term_single_process(x_gene, importance_type)

    def explain_samples_term_single_process(self, x_gene, importance_type):
        """获取预测数据集中每一个样本决策过程中的增强特征的特征重要性

        Parameters
        ----------
        importance_type: str, must be'shap'

        Returns
        ------
        importance_df: DataFrame, shape=(n_terms, n_samples)
            index: term_name
        
        importance_weight_df: DataFrame, shape=(n_terms, n_samples), index is term_name,
            weighted importance
        """
        import time
        nob_dict, lcb_array = self._get_bc(x_gene)
        term_feature_names = [
            term for layer in self.model.term_layer_list for term in layer
        ]
        n_smaples = np.shape(x_gene)[0]
        n_terms = len(term_feature_names)
        unweight_df = pd.DataFrame(np.zeros((n_smaples, n_terms)), columns=term_feature_names)
        weight_df = pd.DataFrame(np.zeros((n_smaples, n_terms)), columns=term_feature_names)

        # 获取结点权重
        weights = self.model.weighter.weights * 1092
        # 获取结点的父亲数量
        term_child_map = self.model.term_child_map_used 
        term_nparent_map = {}
        for children in term_child_map.values():
            for term_name in children:
                if term_nparent_map.get(term_name):
                    term_nparent_map[term_name] += 1
                else:
                    term_nparent_map[term_name] = 1
        
        # 逐层解释
        for layer_id, layer in zip(np.arange(6)[::-1], self.term_layer_list[::-1]):
            
            start_time = time.time()
            for term_name in layer:
                x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
                node_weight = float(weights.loc[term_name])
                # 选择解释方案
                if importance_type == 'shap':
                    imptc_df = self._explain_predict_input_single_process(x, term_name)     # 单进程
                else:
                    assert False, "only 'shap' can passed!"
                    pass
                # 筛选GO的重要性
                all_term_columns = imptc_df.columns.values[
                    list(map(lambda idx: "GO" in idx, imptc_df.columns.values))
                ]
                term_imptc_df = imptc_df.loc[:, all_term_columns]  # 仅保留term的特征重要性
                term_imptc_df /= term_imptc_df.sum(axis=1).to_numpy().reshape((-1,1)) # 和为1  
                # 修正的重要性计算方法
                unweight_df.loc[:, term_imptc_df.columns] = \
                    term_imptc_df * term_imptc_df.shape[1]
                weight_df.loc[:, term_imptc_df.columns] = \
                    term_imptc_df * node_weight * term_imptc_df.shape[1]

            end_time = time.time()
            print(f"layer{layer_id} finished, elapsed {end_time-start_time:.2f}s")
        
        # 加权的除上每个节点的父亲数量
        for term_name, n_parent in term_nparent_map.items():
            unweight_df[term_name] /= n_parent
            weight_df[term_name] /= n_parent

        importance_df = unweight_df
        importance_weight_df = weight_df
        return importance_df.T, importance_weight_df.T        

    def extract_rule(self, term_name:str, x_gene, weight=False):
        """抽取决策规则
        
        Parameters
        ----------
        term_name: str
        x_gene: gene features
        weight: bool
            if weight is False, the weights of forests are all equal"""
        nob_dict, lcb_array = self._get_bc(x_gene)
        node: NodeRR = self.model._get_node(term_name)
        if weight:
            weights = [0.2 for i in range(5)]
        else:
            weights = [0.2 for i in range(5)]
        x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
        rule_extractor = NodeRRExtractor(
            wrapper_models=node.estimators,
            feature_names=node.feature_names,
            weights=weights,
            x=x,
        )
        ruleset = rule_extractor.get_rules_string(x)
        return ruleset

    def extract_rule_with_bc(self, nob_dict, lcb_array, term_name: str, x_gene, weight=False):
        """加速的版本的extract_rule, 要求传入增强特征, 以达到加速的目的"""
        node: NodeRR = self.model._get_node(term_name)
        if weight:
            weights = [0.2 for i in range(5)]
        else:
            weights = [0.2 for i in range(5)]
        x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
        rule_extractor = NodeRRExtractor(
            wrapper_models=node.estimators,
            feature_names=node.feature_names,
            weights=weights,
            x=x,
        )
        ruleset = rule_extractor.get_rules_string(x)
        return ruleset

    def explain_node(self, x_gene, term_name):
        """获取某个node的特征重要性, 解释样本
        
        Returns
        -------
        importance_df: DataFrame. shape=(n_features, n_samples)
            index: feature_names, colunms: sample_id
        """
        nob_dict, lcb_array = self._get_bc(x_gene)
        x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
        return self._explain_predict_input_single_process(x, term_name).T

    def get_shapley_values(self, x_gene):
        """获取全局的shapley值, 包括基因特征和term原始增强特征
        
        Returns
        -------
        gene_shapley_values: DataFrame, shape = (n_samples, n_genes)
        term_shapley_values: DataFrame, shpae = (n_samples,)
        """
        nob_dict, lcb_array = self._get_bc(x_gene)
        n_smaples, n_genes = np.shape(x_gene)
        gene_feature_names = [
            f"Drug{drug_id}_gene{gene_id}" for drug_id in ['A', 'B'] for gene_id in range(int(n_genes/2))
        ]
        term_feature_names = [
            term for layer in self.model.term_layer_list for term in layer
        ]
        n_terms = len(term_feature_names)
        from operator import itemgetter
        term_features = np.squeeze(itemgetter(*term_feature_names)(nob_dict)).T
        pd.DataFrame(term_features, columns=term_feature_names).to_csv("/home/tq/project/drugcomb_V1216/全局shapley值_bliss/term_features.csv")
        # 获取结点权重
        weights = self.model.weighter.weights * 1092

        # 初始化shapley值DataFrame
        gene_shapley_values = pd.DataFrame(np.zeros_like(x_gene), columns=gene_feature_names)
        term_shapley_values = pd.DataFrame(np.zeros((n_smaples, n_terms)), columns=term_feature_names)
        base_values = pd.DataFrame(index=weights.index, columns=['base_value'])
        # 逐层解释
        for layer_id, layer in zip(np.arange(6)[::-1], self.term_layer_list[::-1]):
            # 逐结点解释
            for term_name in layer:
                x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
                node_weight = float(weights.loc[term_name])
                explaintion = self._shap_explain(x, term_name)
                base_values.loc[term_name, 'base_value'] = explaintion.base_values
                shapley_values = pd.DataFrame(explaintion.values, columns=explaintion.feature_names)
                gene_cols = shapley_values.columns[shapley_values.columns.str.contains('gene')]
                gene_shapley_values.loc[:, gene_cols] += shapley_values.loc[:, gene_cols]*node_weight
                term_cols = shapley_values.columns[shapley_values.columns.str.contains('GO')]
                term_shapley_values.loc[:, term_cols] += shapley_values.loc[:, term_cols]*node_weight
        base_value_ave = base_values.sum()/1092
        gene_explaintion = shap.Explanation(
            values=gene_shapley_values.values/1092, 
            base_values=base_value_ave, 
            data=x_gene, 
            feature_names=gene_feature_names
            )

        term_explaintion = shap.Explanation(
            values=term_shapley_values.values/1092,
            base_values=base_value_ave,
            data=term_features,
            feature_names=term_feature_names
        )
        # return gene_shapley_values, term_shapley_values
        return gene_explaintion, term_explaintion

    def _shap_explain(self, x, term_name):
        """获取样本在单个节点中的Shapley values
        
        Returns
        -------
        node_explaintion: shap.Explaintion

        """
        node: NodeRR = self.model._get_node(term_name)
        feature_names = node.feature_names
        explanations = []

        for forest_id, model in enumerate(node.estimators):
            explainer = shap.explainers.Tree(
                model=model, 
                feature_perturbation="tree_path_dependent", 
                feature_names=feature_names,
            )
            # temp = explainer(x)
            explanations.append(explainer(x))
        # 取平均值
        shapley_values_ave = np.mean(
            [explaintion.values for explaintion in list(explanations)],
            axis=0,
        )    # explaintion's shape=(n_samples, n_features)
        base_value_ave = np.mean(
            [explaintion.base_values for explaintion in explanations], 
            axis=0
        )
        node_explaintion = shap.Explanation(
            values=shapley_values_ave, 
            base_values=base_value_ave, 
            data=x, 
            feature_names=feature_names
            )
        return node_explaintion

    def _explain_predict_input_single_process(self, x, term_name):
        """单线程解释样本在单个节点中样本的特征重要性
                
        Returns
        ------
            importance_df: DataFrame, shape: (n_samples, n_features)
        """
        node_explaintion = self._shap_explain(x, term_name)
        importance = self._calcul_feature_importance_multi_samples(node_explaintion)
        importance_df = pd.DataFrame(importance, columns=node_explaintion.feature_names)

        return importance_df

    @staticmethod
    def _calcul_feature_importance(shapley_value: shap.Explanation):
        """利用给定的shapley_value以计算总和特征重要性
        
        Parameters
        ----------
        shapley_value: shap.Explanation
        
        Returns 
        -------
        importance: ndarray
            特征重要性的数组
        """
        assert isinstance(shapley_value, shap.Explanation), \
            "参数shapley_value必须是shap.Explanation类型"
        total_shapley_values = np.sum(np.abs(shapley_value.values), axis=0)
        importance = total_shapley_values / np.sum(total_shapley_values)
        return importance

    @staticmethod
    def _calcul_feature_importance_multi_samples(shapley_value: shap.Explanation):
        """利用给定的shapley_value以计算各个样本的特征重要性
        
        Parameters
        ----------
        shapley_value: shap.Explanation
        
        Returns 
        ------
        importance: ndarray
        """  
        assert isinstance(shapley_value, shap.Explanation), \
            "参数shapley_value必须是shap.Explanation类型"
        total_shapley_values = np.sum(np.abs(shapley_value.values), axis=1)
        importance = np.abs(shapley_value.values) / total_shapley_values.reshape((-1, 1))
        # importance = np.transpose(np.abs(shapley_value.values).T / total_shapley_values)
        # print(np.sum(importance, axis=1))
        return importance

    @classmethod
    def _get_important_nodes_list(
        cls, 
        importance_df: pd.DataFrame, 
        importance_method: str, 
        threshold
        ):
        """获取重要的nodes, 返回重要nodes的list
        
        Parameters
        ----------
        importance_df: 特征重要性的DataFrame
        importance_method: 0 or 1
        
        Returns
        -------
        important_nodes_list: list
            重要性得分较高的list
        """
        if cls._check_leaf_node(importance_df): return []
        go_index = importance_df.index.values[
            list(map(lambda idx: "GO" in idx, importance_df.index.values))
        ]
        nodes_importance_df = importance_df.loc[go_index]   # 仅保留GO的特征重要性
        nodes_importance_df /= nodes_importance_df.sum()
        important_nodes_list = cls._important_metric(
                                    nodes_importance_df, 
                                    importance_method, 
                                    threshold
                                    )
        return important_nodes_list

    @staticmethod
    def _check_leaf_node(importance_df):
        """检查当前的importance_df是否是属于叶节点"""
        index_list = np.array(importance_df.index, dtype=str)
        return np.all(np.char.rfind(index_list, "GO")==-1)
        
    @staticmethod
    def _important_metric(features_importance_df, method, threshold=0.5):
        """在一批标有重要性得分的节点中筛选出重要节点的策略
        parameter
        ---------
            method: int, 0 or 1
                method = 0: 利用数量级作为筛选方案
                method = 1: 对重要性得分排序后, 从最高得分结点开始选择, 逐个加入选择集合, 
                直到集合中的总得分超过设置的阈值
        
        return
        ------
            important_features_list
        """
        imptc_df_sorted = features_importance_df.sort_values(
                                                'importance_score', 
                                                ascending=False,
                                                )
        important_features_list = []

        if method == 0:
            def get_decimal_place(decimal: float):
                for i in range(1, 7):
                    if decimal * np.power(10,i) > 1:
                        return i 
            max_place = get_decimal_place(float(features_importance_df.max()))
            imptc_df_sorted = imptc_df_sorted * np.power(10, max_place)
            important_features_list = imptc_df_sorted.loc[\
                imptc_df_sorted['importance_score']>=1].index
        elif method == 1:
            assert threshold > 0 and threshold < 1, "如果method=1, 则threshold必须合法"
            imptc_df_sorted_scalered = imptc_df_sorted / imptc_df_sorted.sum()
            sum_score = 0
            important_features_list = []
            for term_name, impct_score in imptc_df_sorted_scalered.iterrows():
                sum_score += impct_score.values[0]
                important_features_list.append(term_name)
                if sum_score > threshold: break
        
        return important_features_list

class GOFCLFExplainer(GOFRExplainer): 
    def __init__(
        self, 
        GOFR: GOForestClassifier, 
        ):
        self.model: GOForestClassifier = GOFR
        self.log_dir = GOFR.log_dir
        self.root = GOFR.root
        self.gene_dim = GOFR.gene_dim
        self.term_layer_list = GOFR.term_layer_list
        self.n_layers = GOFR.n_layers
        self.term_layer_map = GOFR.term_layer_map

        # 获取结点的父亲数量
        term_child_map = self.model.term_child_map_used 
        term_nparent_map = {}
        for children in term_child_map.values():
            for term_name in children:
                if term_nparent_map.get(term_name):
                    term_nparent_map[term_name] += 1
                else:
                    term_nparent_map[term_name] = 1
        self.term_nparent_map = term_nparent_map

    def _get_bc(self, x_gene):
        """bc"""
        lcb_dims = []
        self.model.predict(x_gene)
        nob_dict = self.model.nob_dict
        lcb_dict = self.model.lcb_dict
        lcb_array = []
        for key, value in lcb_dict.items():
            lcb_dims.append(np.shape(value)[1])
            # lcb_array.extend(value)
            if len(lcb_array) == 0:
                lcb_array = value
            else:
                lcb_array = np.hstack((lcb_array, value))
        # lcb_array = np.transpose(np.squeeze(lcb_array))
        self.lcb_dims = lcb_dims
        return nob_dict, lcb_array
        
    def _build_node_input(self, term_name, x_gene, nob_dict, lcb_array):
        node: NodeCLF = self.model._get_node(term_name)
        x = x_gene[:, node.gene_idx]
        lcb_array = lcb_array[:, :2*(node.layer_id-2)] if node.layer_id >= 2 else []
        node_input = node._process_features('predict', x, nob_dict, lcb_array)
        return node_input
    
    def _merge_feature_names(self, feature_names):
        
        feature_names = np.array(feature_names).astype(str)
        term_mask = np.char.find(feature_names, 'GO') >= 0
        term_list = feature_names[term_mask]
        term_list = np.unique([item[0] for item in np.char.split(term_list, sep='_')])
        n_gene = np.sum(np.char.find(feature_names, 'gene') >= 0)
        n_nob = np.sum(term_mask)
        feature_names_merged = np.hstack([
            feature_names[:n_gene], term_list, feature_names[n_gene+n_nob:]])
        return feature_names_merged, n_gene, n_nob

    def _explain_nodes_multi_process(self, term_name, x):
        """解释结点的整体知识, 获取重要性得分, x
        
        Parameters
        ----------
        term_name: str
        x: input for term
        
        Returns
        -------
        importance_df: DataFrame, shape=(number of node's features, 1)
            index: feature names, columns='importance_score'
        """
        from multiprocessing import Process, Manager
        shapley_values_dict = {}
        node: NodeCLF = self.model._get_node(term_name)
        feature_names = node.feature_names

        def get_shap(forest_id, res_dict):
            model = node.estimators[forest_id]
            # make explainer
            explainer = shap.explainers.Tree(model, x, feature_names=feature_names)
            # get shap
            explanation: shap.Explanation = explainer(x,  check_additivity=False )
            res_dict[str(forest_id)] = explanation
        
        # 多进程
        manager = Manager()
        res_dict = manager.dict()
        process_list = [Process(target=get_shap, args=(i, res_dict)) for i in range(5)]
        for process in process_list:
            process.start()
        for process in process_list:
            process.join()
        # 处理shapley_values
        explanantions = [res_dict.get(str(i)) for i in range(5)]
        importances = np.sum([self._calcul_feature_importance(explanation) 
            for explanation in explanantions], axis=0)
        importances = importances / np.sum(importances)
        # 处理合并的names
        feature_names_merged, _, _ = self._merge_feature_names(feature_names)
        importance_df = pd.DataFrame(
            importances, 
            index=feature_names_merged, 
            columns=['importance_score']
        )
        # importance_df = self._merge_term_class_importance(importance_df, node.child_list)
        
        return importance_df

    def _calcul_feature_importance(self, explanation: shap.Explanation):
        """利用给定的shapley_value以计算总和特征重要性
        
        Parameters
        ----------
        explanation: shap.Explanation
        
        Returns 
        -------
        importance: ndarray
            特征重要性的数组
        """
        assert isinstance(explanation, shap.Explanation), \
            "参数explanation必须是shap.Explanation类型"
        # 合并terms的Shapley_values
        explanation = self._merge_term_class_shapley(explanation)
        total_shapley_values = np.sum(np.abs(explanation.values), axis=0)
        importance = total_shapley_values / np.sum(total_shapley_values)
        return importance

    def _get_xgb_importance(self, term_name, importance_type='gain'):
        importance_df = super()._get_xgb_importance(term_name, importance_type)
        node: NodeCLF = self.model._get_node(term_name)
        importance_df = self._merge_term_class_importance(importance_df, node.child_list)
        return importance_df

    def explain_DFS(
        self, 
        x_gene, 
        importance_type, 
        importance_method, 
        threshold=0.5
    ):
        """从输出层出发搜索重要Term"""
        return super().explain_DFS(x_gene, importance_type, importance_method, threshold)

    def get_imptc_path_DFS(self, term_impt_nodes_dict: dict):
        return super().get_imptc_path_DFS(term_impt_nodes_dict)
    
    def explain_hierarchy(self, x_gene, importance_type, importance_method, threshold):
        """直接使用每层的节点输出作为特征, 分层找当前层的top重要的节点

        Parameters
        ----------
        importance_type: str, 'shap', 'gain', 'cover' or 'weight'
        importance_method: 0 or 1
            method = 0: 利用数量级作为筛选方案
            method = 1: 对重要性得分排序后, 从最高得分结点开始选择, 逐个加入选择集合, 
            直到集合中的总得分超过设置的阈值
        threshold: float, 0 < threshold < 1

        Returns
        ------
        layer_imptc_list: list, shape=(n_layers, )
            所有层的结点的重要性得分. 
        """
        nob_dict, lcb_array = self._get_bc(x_gene)
        from multiprocessing import Process, Manager
        import time
        # initial the dictionary
        imptc_dict = Manager().dict(
            {term : 0.0 for layer in self.term_layer_list for term in layer})
        weighted_imptc_dict = Manager().dict({
            term : 0.0 for layer in self.term_layer_list for term in layer})
        # 获取结点权重
        weights = self.model.weighter.weights * len(self.model.term_direct_gene_map)
        
        # define update function
        def update_dict(nodes_imptc_df: DataFrame):
            for term_name, importance in nodes_imptc_df.iterrows():
                if "GO" in term_name:
                    imptc_dict[term_name] += float(importance) * len(nodes_imptc_df)
        def update_weight_dict(nodes_imptc_df: DataFrame, node_weight: float):
            for term_name, importance in nodes_imptc_df.iterrows():
                if "GO" in term_name:
                    # weighted_imptc_dict[term_name] += float(importance) * node_weight
                    weighted_imptc_dict[term_name] += \
                        float(importance) * node_weight * len(nodes_imptc_df)
        def _sub_process(layer_id, layer):
            """逐层解释的子线程函数"""
            start_time = time.time()
            for term_name in layer:
                x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
                node_weight = float(weights.loc[term_name])
                # 选择解释方案
                if importance_type == 'shap':
                    imptc_df = self._explain_nodes_multi_process(term_name, x)    # 多进程
                else:
                    imptc_df = self._get_xgb_importance(term_name, importance_type)
                    pass
                # 过滤非Gene的重要性
                go_index = imptc_df.index.values[
                    list(map(lambda idx: "GO" in idx, imptc_df.index.values))   
                ]
                nodes_imptc_df = imptc_df.loc[go_index]   # 仅保留GO的特征重要性
                update_dict(nodes_imptc_df)
                update_weight_dict(nodes_imptc_df, node_weight)
            end_time = time.time()
            print(f"layer{layer_id} finished, elapsed {end_time-start_time:.2f}s")

        # 多线程解释
        process_list = [
            Process(target=_sub_process, args=(layer_id, layer,)) 
            for layer_id, layer in zip(np.arange(6)[::-1] ,self.term_layer_list[::-1])
        ]
        for process in process_list:
            process.start()
        for process in process_list:
            process.join()

        # 加权的除上每个节点的父亲数量
        for term_name, n_parent in self.term_nparent_map.items():
            weighted_imptc_dict[term_name] /= n_parent
            imptc_dict[term_name] /= n_parent

        # 将字典分层, 分出每一层的重要节点
        from operator import itemgetter
        layer_imptc_list = []
        layer_weight_imptc_list = []
        for layer_id, layer in enumerate(self.term_layer_list):
            df = pd.DataFrame(
                itemgetter(*layer)(imptc_dict), 
                index=layer, 
                columns=["importance_score"]
            )
            weight_df = pd.DataFrame(
                itemgetter(*layer)(weighted_imptc_dict), 
                index=layer, 
                columns=["importance_score"]
            )
            important_nodes = self._important_metric(
                                    df, 
                                    method=importance_method, 
                                    threshold=threshold
                                    )
            weight_important_nodes = self._important_metric(
                weight_df, 
                method=importance_method,
                threshold=threshold
            )
            df['is important'] = pd.DataFrame(
                [1 if term_name in important_nodes else 0 for term_name in layer], 
                index=layer
            )
            weight_df['is important'] = pd.DataFrame(
                [1 if term_name in weight_important_nodes else 0 for term_name in layer], 
                index=layer
            )
            layer_imptc_list.append(df)
            layer_weight_imptc_list.append(weight_df)
        return layer_imptc_list, layer_weight_imptc_list
    
    def explain_genes(
        self, 
        x_gene, 
        importance_type, 
    ):
        """获得模型的基因特征重要性
      
        Parameters
        ----------
        x_gene
        importance_type: str, 'shap', 'gain', 'cover' or 'weight'

        Returns
        ------
        importance_df: DataFrame
            index: term_name
        """
        return super().explain_genes(
            x_gene, 
            importance_type, 
        )

    def explain_samples_gene(self, x_gene, importance_type):
        """计算样本集的gene特征重要性"""
        return super().explain_samples_gene(x_gene, importance_type)

    def explain_samples_term(self, x_gene, importance_type):
        """计算样本集的term重要性"""
        return self.explain_samples_term_single_process(x_gene, importance_type)

    def explain_samples_term_single_process(self, x_gene, importance_type):
        """单进程计算样本集的term重要性"""
        importance_df, importance_weight_df = super().explain_samples_term_single_process(
            x_gene, 
            importance_type,
        )
        return importance_df, importance_weight_df
    
    def _explain_predict_input_single_process(self, x, term_name):
        """分类模型的单进程解释样本的重要性
        Patameters:
        x: array
        term_name: str
        
        Returns
        -------
        importance_df: DataFrame, shape=(n_samples, n_features)
        """
        node_explaintion = self._shap_explain(x, term_name)
        feature_names_merged,_ ,_ = self._merge_feature_names(node_explaintion.feature_names)
        importance = self._calcul_feature_importance_multi_samples(node_explaintion)
        importance_df = pd.DataFrame(importance, columns=feature_names_merged)

        return importance_df

    def _calcul_feature_importance_multi_samples(self, explanation: shap.Explanation):
        """利用给定的shapley_value以计算各个样本的特征重要性
        
        Parameters
        ----------
        shapley_value: shap.explanation
        
        Returns 
        ------
        importance: ndarray
        """  
        assert isinstance(explanation, shap.Explanation), \
            "参数shapley_value必须是shap.Explanation类型"
        # 先合并nob的shapley值
        explanation = self._merge_term_class_shapley(explanation)
        total_shapley_values = np.sum(np.abs(explanation.values), axis=1)
        importance = np.abs(explanation.values) / total_shapley_values.reshape((-1, 1))

        return importance


    def extract_rule(self, term_name: str, x_gene, weight=False):
        """抽取决策规则
        
        Parameters
        ----------
        term_name: str
        x_gene: gene features
        weight: bool
            if weight is False, the weights of forests are all equal
        """
        nob_dict, lcb_array = self._get_bc(x_gene)
        node: NodeCLF = self.model._get_node(term_name)
        if weight:
            weights = [0.2 for i in range(5)]
        else:
            weights = [0.2 for i in range(5)]
        x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
        rule_extractor = NodeCLFExtractor(
            wrapper_models=node.estimators,
            feature_names=node.feature_names,
            weights=weights,
            x=x,
        )
        ruleset = rule_extractor.get_rules_string(x)
        return ruleset
    
    def extract_rule_with_bc(self, nob_dict, lcb_array, term_name: str, x_gene, weight=False):
        """加速的版本的extract_rule"""
        node: NodeCLF = self.model._get_node(term_name)
        if weight:
            weights = [0.2 for i in range(5)]
        else:
            weights = [0.2 for i in range(5)]
        x = self._build_node_input(term_name, x_gene, nob_dict, lcb_array)
        rule_extractor = NodeCLFExtractor(
            wrapper_models=node.estimators,
            feature_names=node.feature_names,
            weights=weights,
            x=x,
        )
        ruleset = rule_extractor.get_rules_string(x)
        return ruleset

    def explain_node(self, x_gene, term_name):
        """提取单个node的解释"""
        importance_df = super().explain_node(x_gene, term_name)
        node: NodeCLF = self.model._get_node(term_name)
        return self._merge_term_class_importance(importance_df, node.child_list)
    
    def _merge_term_class_shapley(self, explanation: shap.Explanation):
        """合并原始增强特征的term的Shapley values"""
        def sum_by_multicol(array, step=2):
            # 两列两列地相加
            n_row, n_col = array.shape[0], int(array.shape[1]/step)
            result = np.empty((n_row, n_col))
            for i in range(n_col):
                result[:, i] = np.sum(array[:, i*step:(i+1)*step])
            return result
        feature_names = explanation.feature_names
        feature_names_merged, n_gene, n_nob = self._merge_feature_names(feature_names)

        shapley_values = explanation.values
        term_shapley_values = sum_by_multicol(shapley_values[:, n_gene:n_gene+n_nob])
        shapley_values_merged = np.hstack([
            shapley_values[:,:n_gene],
            term_shapley_values,
            shapley_values[:,n_gene+n_nob:],
        ])
        explanation_merged = shap.Explanation(
            values=shapley_values_merged,
            base_values=explanation.base_values,
            feature_names=feature_names_merged,
        )
        return explanation_merged


    @staticmethod
    def _merge_term_class_importance(importance_df: DataFrame, term_list: list):
        """合并增强特征的特征重要性(将term_0_class, term_1_class的重要性合并为一个term的重要性

        Parameters
        ----------
        importance_df: DataFrame
        term_list: list
        """
        # 合并 nob(class0, class1) 的特征cv_5重要性
        term_importance_list = []
        for term_name in term_list:
            # term_imptc = importance_df.loc[importance_df.index.str.contains(term_name)].sum(axis=0)[0]
            term_imptc = importance_df.loc[importance_df.index.str.contains(term_name)].sum(axis=0)
            term_importance_list.append(term_imptc)
        # if len(term_importance_list) == 0:
        #     term_importance_list = [[]]*importance_df.shape[1]

        # 重新拼接gene, nob, ncb, lcb的重要性
        n_gene = len(importance_df.loc[importance_df.index.str.contains("gene")])
        n_term = len(term_list)
        importance_value = importance_df.values
        if len(term_importance_list) != 0:
            importances = np.concatenate(
                [
                    importance_value[0:n_gene], 
                    term_importance_list, 
                    importance_value[n_gene+n_term*2: ]
                ],
                axis=0,
            )
        else:
            importances = np.concatenate(
                [
                    importance_value[0:n_gene], 
                    importance_value[n_gene+n_term*2: ]
                ],
                axis=0,
            )
        feature_names = importance_df.index
        index = np.concatenate(
            [feature_names[0:n_gene], term_list, feature_names[n_gene+n_term*2: ]],
            axis=0,
        )  
        columns = importance_df.columns
        return pd.DataFrame(importances, index=index, columns=columns)
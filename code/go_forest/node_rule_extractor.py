""" 为节点中的五折交叉验证的森林抽取对应的决策路径

"""
import numpy as np
from .rule_extractor import XGBRRExtractor, simple_rules_reg
from .rule_extractor import XGBCLFExtractor, simple_rules_binary
from typing import List, Sequence
from xgboost import XGBModel, DMatrix
from .triple import TripleSet, Rule, Antecedent, Consequent

class NodeRRExtractor:
    def __init__(
        self, 
        wrapper_models: Sequence[XGBModel],
        feature_names: list = None,
        weights: Sequence = None,
        x: np.array = None,
        ) -> None:
        """
        x: 待解释的样本特征集
        """
        self.wrapper_models: Sequence[XGBModel] = wrapper_models
        self.feature_names: list = feature_names
        self.weights: Sequence = weights
        # self.base_scores = [model.base_score for model in wrapper_models]
        self.base_scores = [0.5 for model in wrapper_models]
        self.extractors: Sequence[XGBRRExtractor] = \
            [XGBRRExtractor(xgb, self.feature_names) for xgb in wrapper_models]
        self.feature_importance = self._get_feature_importance(x)
    
    def _update_importance(self, x: np.ndarray, importance: list) -> np.ndarray:
        """使用shapley值更新特征重要性"""
        importance_shap = 0
        for extractor in self.extractors:
            importance_shap += extractor.get_score_shap(x)
        importance_shap /= np.sum(importance_shap)
        feature_importance = importance_shap + importance
        feature_importance /= np.sum(feature_importance)
        return feature_importance

    def _get_feature_importance(self, x: np.ndarray=None) -> np.ndarray:
        """获得基于weight和cover的特征重要性"""
        importance = 0
        for xgb_model in self.wrapper_models:
            for importance_type in ["weight", "cover"]:
                xgb_model.importance_type = importance_type
                importance += xgb_model.feature_importances_
        importance /= np.sum(importance)
        if x is not None:
            return self._update_importance(x, importance)
        else:
            return importance
   

    def _get_triples(self, x):
        """提取x中每个样本的决策路径三元组
        Parameters
        ---------
        x : ndarray, samples

        Returns
        -------
        triple_sets: array_like, shape = (n_samples, n_xgbs*n_estimators), 
            the type of objects in triple_sets is Rule.triples
        """
        triple_sets = np.empty((np.shape(x)[0], 0)) # shape=(n_samples, 0)
        data = DMatrix(x, feature_names = self.feature_names)
        for extractor in self.extractors:
            leaf_pred = extractor.Booster.predict(data, pred_leaf=True)
            if len(triple_sets)==0:
                triple_sets = extractor._extract_triples(leaf_pred)
            else:
                # temp = extractor._extract_rules(leaf_pred)
                triple_sets = np.concatenate(
                    (triple_sets, extractor._extract_triples(leaf_pred)),
                    axis=1
                )
                # for triples1, triples2 in zip(rules, temp):
                #     triples1.extend(triples2) 
        # print(rules.shape)
        # return np.array(rules, dtype='O')
        return triple_sets
        
    def get_rules_string(self, x):
        """对x进行解释. 抽取, 合并, 修剪合适的规则, 以形成解释. 

        """
        y_preds = np.sum(
            np.multiply(
                np.transpose([xgb.predict(x) for xgb in self.wrapper_models]),
                self.weights,
                ), 
            axis=1
        )
        rules = self._get_triples(x)
        rules_string = []
        base_score = np.sum(np.multiply(self.weights, self.base_scores))
        for rule_triples, y_pred in zip(rules, y_preds):
            rule_twins = self._simple_rules(rule_triples, y_pred)
            rule_twins = [self._prune_rule(rule) for rule in rule_twins]
            rules_string.append([rule.stringify() for rule in rule_twins])
        return rules_string
    
    def _simple_rules(self, triplesets: Sequence[TripleSet], y_pred):
        """从ruleset中抽取适量的规则以组成解释
        
        Parameters
        ----------
        triples: list, 规则(由三元组组成的)集合
        y_pred: float, 单个样本的预测值

        Returns 
        -------
        rule: Rule, 最后合并的解释. 
            solutions: 选中的规则集合
        """
        def select(array):
            """选择最能后件填补gap的规则"""
            gap = y_pred-value_addup
            # 根据预测值的正负选择不同的逻辑
            if y_pred > 0:
                mask = np.array(value_list<gap) & np.array(value_list>0)
                if mask.any() == False:
                    return None
                mask_True_index = np.argwhere(mask).reshape(-1)
                index = mask_True_index[np.argmax(array[mask])]
            else:
                mask = np.array(value_list>gap) & np.array(value_list<0)
                if mask.any() == False:
                    return None
                mask_True_index = np.argwhere(mask).reshape(-1)
                index = mask_True_index[np.argmin(array[mask])]
            return index
        # 置空solutions
        solutions = []
        # 初始化上下界
        feature_boundary = np.array(
            [[np.inf for i in self.feature_names],
            [-np.inf for i in self.feature_names]]
        )
        value_addup = 0
        value_list = np.array([triples.values[-1][-1] for triples in triplesets])
        # 选择合适的triples加入solutions
        while((y_pred>0 and value_addup<y_pred) or (y_pred<0 and value_addup>y_pred)):
            index = select(value_list)
            if index is not None:              
                # 如果index指向的rule与当前的boundary冲突
                if self._check_contradiciton(feature_boundary, Rule(triplesets[index])):
                    value_list = np.delete(value_list, index)
                    triplesets = np.delete(triplesets, index)
                    continue
                # 如果没有冲突， 则更新solutions, boundary等
                value_addup += value_list[index]
                solutions.append(triplesets[index])
                self._update_boundary(feature_boundary, Rule(triplesets[index]))
                value_list = np.delete(value_list, index)
                triplesets = np.delete(triplesets, index)
            if index is None: 
                # 如果一条规则都不满足， 则选择后件结果最接近y_pred的规则
                if len(solutions) == 0:
                    index = np.argmax(value_list)
                    solutions.append(triplesets[index])
                break
        rule_twins = simple_rules_reg(solutions, self.feature_names, self.base_scores)
        return rule_twins

    def _prune_rule(self, triples: Rule):
        """修简规则, 保留比较重要的特征的前件"""
        def _filter_index_bc() -> List:
            """过滤掉增强特征的前件, 只返回基因特征的前件"""
            index_features = triples.index_antecedent
            # np.argsort() 默认是升序排序
            index_antecedent_sorted = np.argsort(self.feature_importance[index_features])[::-1] # 比如说, 从100个特征抽取10个出来,这里的排序结果是在0到10之间
            index_features_sorted = index_features[index_antecedent_sorted]                     # 这个才是 特征子集(10个特征) 对应到0~100的索引
            
            index_selected = []
            for idx_antecedent, idx_feature in zip(index_antecedent_sorted, index_features_sorted):
                name = self.feature_names[idx_feature] 
                if "gene" in name or "GO" in name:
                    index_selected.append(idx_antecedent)
                if len(index_selected) >= 10:
                    break
            return index_selected
        if triples.is_empty():
            return triples
        index = _filter_index_bc()
        triples_new = [triples.antecedents[i] for i in index]
        triples_new.append(triples.consequent)
        rule_new = Rule(triples_new, feature_names=triples.feaure_names)
        return rule_new

    @staticmethod
    def _check_contradiciton(boundary, rule: Rule):
        """检查rule是否和给定的boundary冲突, 返回True则表示冲突, False表示不冲突. 
        """
        contrary = []
        for ant in rule.antecedents:
            if (ant[1] == 0 and ant[2] <= boundary[1][ant[0]]) or \
                (ant[1] == 1 and ant[2] >= boundary[0][ant[0]]):
                contrary.append(ant[0])
        return len(contrary) != 0
    @staticmethod
    def _update_boundary(boundary, rule: Rule):
        """更新规则子集的boundary"""
        for ant in rule.antecedents:
            if ant[1] == 0 and ant[2] < boundary[0][ant[0]]:
                boundary[0][ant.feature_index] = ant.threshold
            elif ant[1] == 1 and ant[2] > boundary[1][ant[0]]:
                boundary[1][ant.feature_index] = ant.threshold
        

class NodeCLFExtractor(NodeRRExtractor):
    def __init__(
        self,
        wrapper_models: Sequence[XGBModel],
        feature_names: list = None,
        weights: Sequence = None,
        x: np.array = None,
    ) -> None:
        super().__init__(
            wrapper_models,
            feature_names,
            weights,
            x,
        )
        self.extractors: Sequence[XGBCLFExtractor] = [
            XGBCLFExtractor(xgb, self.feature_names)
            for xgb in wrapper_models
        ]
    
    def _simple_rules(self, triples_set: Sequence[TripleSet], y_pred):
        """
        Parameters
        ----------
        triples_set: list[Triples]
        """
        def filter(array, pred_class):
            """从value_list中选出合适的value索引
                y_pred = 0, 搜索出value_list中最小的负数
                y_pred = 1, 搜索出value_list中最大的正数
            """
            if pred_class == 0:
                mask = np.array(value_list < 0)
                if mask.any() == False:
                    return None
                mask_True_index = np.argwhere(mask).reshape(-1)
                index = mask_True_index[np.argmin(array[mask])]

            elif pred_class == 1:
                mask = np.array(value_list > 0)
                if mask.any() == False:
                    return None
                mask_True_index = np.argwhere(mask).reshape(-1)
                index = mask_True_index[np.argmax(array[mask])]
            return index
        solutions = []
        feature_boundary = np.array(
            [[np.inf for i in self.feature_names],
             [-np.inf for i in self.feature_names]]
        )
        margin_value_addup = 0
        value_list = np.array([triples.values[-1][-1] for triples in triples_set])
        if y_pred == 0:
            
            while(margin_value_addup > -10):
                index = filter(value_list, 0)
                if index is not None:
                    # 如果index指向的rule与当前的boundary冲突
                    if self._check_contradiciton(feature_boundary, Rule(triples_set[index])):
                        value_list = np.delete(value_list, index)
                        triples_set = np.delete(triples_set, index)
                        continue
                    # 如果没有冲突， 则更新solutions, boundary等
                    margin_value_addup += value_list[index]
                    solutions.append(triples_set[index])
                    self._update_boundary(feature_boundary, Rule(triples_set[index]))
                    value_list = np.delete(value_list, index)
                    triples_set = np.delete(triples_set, index)
                if index is None: 
                    break
        elif y_pred == 1:
            while(margin_value_addup < 10):
                index = filter(value_list, 1)
                if index is not None:
                    # 如果index指向的rule与当前的boundary冲突
                    if self._check_contradiciton(feature_boundary, Rule(triples_set[index])):
                        value_list = np.delete(value_list, index)
                        triples_set = np.delete(triples_set, index)
                        continue
                    # 如果没有冲突， 则更新solutions, boundary等
                    margin_value_addup += value_list[index]
                    solutions.append(triples_set[index])
                    self._update_boundary(feature_boundary, Rule(triples_set[index]))
                    value_list = np.delete(value_list, index)
                    triples_set = np.delete(triples_set, index)
                if index is None: 
                    break
        # print(f"solutions length: {len(solutions)}")
        return simple_rules_binary(
            y_pred,
            solutions, 
            self.feature_names, 
            self.base_scores
        )

    def _prune_rule(self, triples: Rule):
        """修简规则, 保留比较重要的特征的前件"""
        def _filter_index_bc() -> List:
            """过滤掉增强特征的前件, 只返回基因特征的前件"""
            index_features = triples.index_antecedent
            # np.argsort() 默认是升序排序
            index_antecedent_sorted = np.argsort(self.feature_importance[index_features])[::-1] # 比如说, 从100个特征抽取10个出来,这里的排序结果是在0到10之间
            index_features_sorted = index_features[index_antecedent_sorted]                     # 这个才是 特征子集(10个特征) 对应到0~100的索引
            
            index_selected = []
            for idx_antecedent, idx_feature in zip(index_antecedent_sorted, index_features_sorted):
                name = self.feature_names[idx_feature] 
                if "gene" in name:
                    index_selected.append(idx_antecedent)
                if len(index_selected) >= 10:
                    break
            return index_selected
        if triples.is_empty():
            return triples
        index = _filter_index_bc()
        triples_new = [triples.antecedents[i] for i in index]
        triples_new.append(triples.consequent)
        rule_new = Rule(triples_new, feature_names=triples.feaure_names)
        return rule_new

    def get_rules_string(self, x):
        """对x进行解释. 抽取合并合适的规则, 以形成解释. 

        """
        probas = [xgb.predict_proba(x) for xgb in self.wrapper_models]
        y_preds = np.argmax(
            np.sum(
                [item * weight for item, weight in zip(probas, self.weights)],
                axis=0, 
            ),
            axis=1
        )
        rules = self._get_triples(x)
        rules_string = []
        base_score = np.sum(np.multiply(self.weights, self.base_scores))
        for rule_triples, y_pred in zip(rules, y_preds):
            rule_twins = self._simple_rules(rule_triples, y_pred)
            rule_twins = [self._prune_rule(rule) for rule in rule_twins]
            rules_string.append([rule.stringify() for rule in rule_twins])
        return rules_string




import json
import numpy as np
from xgboost import XGBModel, Booster,DMatrix
from .triple import TripleSet, Antecedent, Consequent, Rule
from typing import Sequence, List

class XGBRRExtractor:
    def __init__(
        self, 
        xgb_model: XGBModel, 
        feature_names = None, 
        weight = None
    ) -> None:
        """
        weight: float, GOForest分配的结点权重        
        """
        self.Booster: Booster = xgb_model.get_booster()
        self.n_estimators: int = xgb_model.n_estimators
        self.base_score: float = xgb_model.base_score
        self.booster_config: dict = \
            json.loads(xgb_model.get_booster().save_raw('json').decode())
        self.feature_names : Sequence[str] = xgb_model.feature_names_in_ \
            if feature_names is None else np.array(feature_names)
        self.gbtrees: dict = \
            self.booster_config["learner"]["gradient_booster"]["model"]
        self.tree_configs: list = self.gbtrees["trees"]
        if weight is not None:
            self.weight = weight
        self.trees_paths: Sequence[Sequence] = self._get_trees_paths()# 所有树的决策路径
        self.trees_rules: Sequence[Sequence[Rule]] = self._get_triples()# 所有树决策路径对应的三元组

    def _extract_node_path(self, leaf, parents):
        """为单个叶节点抽取决策路径"""
        node_list = []
        index = leaf
        while index < len(parents):
            node_list.append(index)
            index = parents[index]
        return np.array(node_list)[::-1]

    def _get_tree_paths(self, tree_config):
        """为单棵树抽取所有的决策路径
        
        Parameters
        ----------
        tree_config : dict
            The config of the tree in xgb 
        
        Returns
        -------
        paths: array_like, List of path for a tree.
            The length of paths is the number of nodes in tree.
            If the i-th node is leaf node, path[i] is like [root, 2, ..., i(leaf)], 
            otherwise, paths[i] = []
        """
        n_nodes = int(tree_config["tree_param"]["num_nodes"])
        is_leaf = np.zeros(n_nodes)
        is_leaf[
            np.array(tree_config["left_children"])==np.array(tree_config["right_children"])
            ] = 1
        parents = tree_config["parents"]
        paths = []
        for i in range(n_nodes):
            if is_leaf[i] == 0:
                paths.append(np.array([]))
            else:
                paths.append(self._extract_node_path(i, parents))
        return paths

    def _get_trees_paths(self):
        """获得xgb森林中所有树的所有路径
        
        Returns
        -------
        trees_paths : list, shape=(n_estimators, )
            The element of trees_paths is a list which saved paths for all node in a tree.
            And this elements come from the method _get_tree_paths(). 
            The path for an internal node is [], and for the leaf node is like
            [root, ..., leaf].
        """
        trees_paths = []
        for tree_config in self.tree_configs:
            trees_paths.append(self._get_tree_paths(tree_config))
        return np.array(trees_paths, dtype='O')
        
    def _convert_to_triple(self, path, tree_config):
        """将path转为存储三元组的列表"""
        if len(path) == 0: return []
        def is_less_than(x) -> bool:
            return int(x % 2)
        triples = []
        for i, node_index in enumerate(path):
            if i < len(path)-1:
                triples.append(Antecedent((
                    int(tree_config["split_indices"][node_index]), 
                    is_less_than(path[i+1]), 
                    tree_config["split_conditions"][node_index], 
                    )))
            else:
                triples.append(Consequent((
                    "value", 
                    2, 
                    tree_config["split_conditions"][node_index],
                    )))
            if hasattr(self, 'weight'):
                triples[-1][-1] *= self.weight
        return Rule(triples, self.feature_names)

    def _get_triples(self):
        """获得xgb森林中所有路径对应的三元组"""
        trees_rules_list = []
        for tree, tree_config in zip(self.trees_paths, self.tree_configs):
            rules = []
            for path in tree:
                rules.append(self._convert_to_triple(path, tree_config))
            trees_rules_list.append(rules)
        return np.array(trees_rules_list, dtype='O')

    def _convert_to_ifelse(self, triple_list):
        """将规则转化成if...else...形式的字符串"""
        rule_str = "if "
        for antecedents in triple_list:
            print(type(antecedents))
            if 'value' not in antecedents:
                rule_str += f"{antecedents} and "
            else:
                rule_str += f"then {antecedents}"
        return rule_str

    def _extract_triples(self, leaf_pred):
        """抽取leaf_pred对应的规则
        
        Parameters
        ----------
        leaf_pred : array_like, shape=(n_samples, n_estimators)
            The predicted leaf index of single sample in each tree.
        
        Returns
        -------
        triple_sets : array_like, shape=(n_samples, n_estimators, ),
            the type of object in triple_sets is Rule.triples
        """
        triples = []  # length=n_smaples
        leaf_pred = np.array(leaf_pred, dtype=np.int32)
        for leaves in leaf_pred:
            path_triple = [self.trees_rules[i][leaf_id].triples 
                            for i, leaf_id in enumerate(leaves)]
            triples.append(path_triple)
        triple_sets = np.array(triples, dtype='O')
        return triple_sets

    def get_rules(self, x) -> Sequence[Rule]:
        """获取样本在xgboost中的决策规则
        
        Parameters
        ----------
        x : array_like, data to predict with. 
            the dimension of x must be 2.

        Returns
        -------
        rules : list, length=(n_samples, 2)
            the first rule lead to positive value or 0 class, 
            the second rule lead to negative value of 1 class
        """
        data = DMatrix(x, feature_names=self.feature_names)
        leaf_pred = self.Booster.predict(data, pred_leaf=True)
        rules = []
        for path_triple in self._extract_triples(leaf_pred):
            rules.append(
                [rule for rule in simple_rules_reg(path_triple, self.feature_names)]
            )
        return np.array(rules, dtype='O')        

    def get_rules_string(self, x):
        """获取样本在xgboost中的决策规则
        
        Parameters
        ----------
        x : array_like, data to predict with. 
            the dimension of x must be 2.

        Returns
        -------
        rules_ifelse : list, length=n_samples
            转化为ifelse之后的规则
        """
        rule_ifesle = []
        # for rule in self.get_rules(x):
        #     rule.stringify
        data = DMatrix(x, feature_names=self.feature_names)
        leaf_pred = self.Booster.predict(data, pred_leaf=True)
        rule_ifesle = []
        for path_triple in self._extract_triples(leaf_pred):
            rule_ifesle.append(
                [rule.stringify() for rule in simple_rules_reg(path_triple, self.feature_names)]
            )
        return np.array(rule_ifesle, dtype='O')

    def get_rules_antecedents(self, x):
        """
        获得对x决策的规则的前件
        
        Returns
        -------
        triples : ndarray, shape=(n_samples, n_estimators)"""
        data = DMatrix(x, feature_names=self.feature_names)
        leaf_pred = self.Booster.predict(data, pred_leaf=True)
        triples = []  # length=n_smaples
        leaf_pred = np.array(leaf_pred, dtype=np.int32)
        for leaves in leaf_pred:
            path_triple = [  # 单个样本在所有基学习器上的决策路径三元组, length=n_estimators
                self.trees_rules[i][leaf_id].antecedents for i, leaf_id in enumerate(leaves)
            ]
            triples.append(path_triple)

        return np.array(triples, dtype='O')

    def get_score_shap(self, x):
        """获取基于shap的特征重要性"""
        import shap
        explainer_shap = shap.explainers.Tree(model=self.Booster)
        shapley_values = explainer_shap.shap_values(x)
        # shapley_values = explainer_shap.values
        score_shap = np.sum(np.abs(shapley_values), axis = 0) 
        score_shap = score_shap / np.sum(score_shap)
        return score_shap

class XGBCLFExtractor(XGBRRExtractor):
    def __init__(
        self, 
        xgb_model: XGBModel, 
        feature_names = None, 
        weight = None
    ) -> None:
        super().__init__(
            xgb_model,
            feature_names,
            weight,
        )
        self.n_class = 2
     
    def _convert_to_triple(self, path, tree_config):
        """将path转为存储三元组的列表"""
        if len(path) == 0: return []
        def is_less_than(x) -> bool:
            return int(x % 2)
        triples = []
        for i, node_index in enumerate(path):
            if i < len(path)-1:
                triples.append(Antecedent((
                    int(tree_config["split_indices"][node_index]), 
                    is_less_than(path[i+1]), 
                    tree_config["split_conditions"][node_index], 
                    )))
            else:
                margin_value = tree_config["split_conditions"][node_index]
                triples.append(Consequent((
                    "value", 
                    2, 
                    margin_value,
                    )))
            if hasattr(self, 'weight'):
                triples[-1][-1] *= self.weight
        return Rule(triples, self.feature_names)


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def merge_rules_reg(ruleset: list, feature_names: Sequence, base_score = None):
    """
    合并规则
    Parameters
    ----------
    ruleset : list
        存储了多个完整规则的路径
    """
    if len(ruleset) == 0: return Rule([])
    final_rule = []
    triple_list = []
    value_list = []
    for rule in ruleset:
        for triple in rule:
            if isinstance(triple, Antecedent):
                triple_list.append(triple)
            elif isinstance(triple, Consequent):
                value_list.append(triple)
    triple_list = np.array(triple_list)
    # feature_indices = np.unique(triple_list[:, 0])
    # 搜索每一个特征的上下限
    for feature_idx in np.unique(triple_list[:, 0]):
        gt_set = triple_list[
            np.array(triple_list[:, 0]==feature_idx) & \
            np.array(triple_list[:, 1]==1)
            ]
        lt_set = triple_list[
            np.array(triple_list[:, 0]==feature_idx) & \
            np.array(triple_list[:, 1]==0)
        ]
        if len(gt_set) != 0:
            lower_limit = np.max(gt_set[:, -1])
            final_rule.append(Antecedent((int(feature_idx), 1, lower_limit)))
        if len(lt_set) != 0:
            upper_limit = np.max(lt_set[:, -1])
            final_rule.append(Antecedent((int(feature_idx), 0, upper_limit)))
        if len(gt_set) != 0 and len(lt_set) != 0 and lower_limit >= upper_limit:
            final_rule.pop(-1)
        if len(gt_set) != 0 and len(lt_set) != 0 and lower_limit >= upper_limit:
            final_rule.pop(-1)
    value_list = np.array(value_list, dtype='O')
    if base_score is not None:
        final_rule.append(Consequent(("value", 2, np.sum(value_list[:, -1]))))
    else:
        final_rule.append(Consequent(("value", 2, np.sum(value_list[:, -1]))))
    rule_merged = Rule(final_rule, feature_names)
    return rule_merged
        
def simple_rules_reg(ruleset: Sequence[TripleSet], feature_names, base_score=None):
    """
    简化不冲突的规则集, 
    为了统一接口, 他会返回两条规则, 分别是预测值大于0和预测值小于0;
    但总有一条是空(None)"""
    positive_set = []
    negative_set = []
    for rule in ruleset:
        rule = rule.values
        if rule[-1][-1] > 0:
            positive_set.append(rule)
        elif rule[-1][-1] < 0:
            negative_set.append(rule)
        else:
            pass

    rule_twins = [merge_rules_reg(positive_set, feature_names, base_score),
                  merge_rules_reg(negative_set, feature_names, base_score)]
    return rule_twins

def merge_rules_binary(
    pred_class, 
    ruleset: list, 
    feature_names: Sequence, 
    base_score=None, 
):
    """二分类规则集的合并
    
    Parameters
    ----------
    pred_class: 0 or 1
    ruleset: list[Rule]
    feature_names: Sequence[str]
    base_score: float

    """
    rule: Rule = merge_rules_reg(ruleset, feature_names, base_score)
    if not hasattr(rule, 'consequent'):
        return rule
    if pred_class == 0:
        rule.consequent[-1] = 1-sigmoid(rule.consequent[-1])
    elif pred_class == 1:
        rule.consequent[-1] = sigmoid(rule.consequent[-1])
    return rule

def simple_rules_binary(
    pred_class, 
    ruleset: Sequence[TripleSet], 
    feature_names, 
    base_score=None
):
    """简化二分类的规则

    Parameters
    ----------
    pred_class: int
    ruleset: Sequence[Triples]

    Returns
    -------
    rules: Sequence[Rule]
        [预测为0类的规则, 预测为1类的规则]
    """
    positive_set = []   # 标准1类的规则集合
    negative_set = []   # 标注0类的规则集合
    for rule in ruleset:
        rule = rule.values
        if rule[-1][-1] > 0:
            positive_set.append(rule)
        elif rule[-1][-1] < 0:
            negative_set.append(rule)
        else:
            pass
    return [
        merge_rules_binary(
            pred_class,
            negative_set, 
            feature_names, 
            base_score
        ),
        merge_rules_binary(
            pred_class, 
            positive_set, 
            feature_names, 
            base_score
        )
    ]




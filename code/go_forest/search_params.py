from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

import numpy as np

def warpper_search_parameters(node, x, y, scorer='accuracy', cv=5):
    """为node中的包裹器搜索最佳的参数
    
    parameters
    ---------- 
        node: Node 

    主要调节的参数: learning_rate, max_depth

    return 
    ------
        params_dict: Node中基学习器的最佳参数
    """
    # est = eval(node.estimator)
    est = node.estimators[0]
    params_grid = {
        "max_depth": [i for i in np.arange(4, 7, 1)],
        # "learning_rate": [i for i in np.arange(0.08, 0.13, 0.1)],
        "n_estimators": [10, 20, 30, 40, 50]
    }
    assert isinstance(est, XGBClassifier) or isinstance(est, XGBRegressor), "classifier must be XGBClassifer or XGBRegressor"
    return grid_search(est, x, y, params_grid, scorer, cv)

def grid_search(est, x, y, params_grid, scorer, cv):
    """为给定的est(学习器)在x, y上做网格搜索, 搜索出最佳的参数

    parameters
    ----------
        est: 基学习器
        x: 特征
        y: 标签
        params_grid: 待搜索的列表

    return
    ------
        best_params: dict, 最佳的参数字典
        best_score: float, 最佳参数对应的分数
    """
    gsearch = GridSearchCV(est, refit='accuracy', param_grid=params_grid, scoring=scorer, cv=cv)
    gsearch.fit(x, y)   
    return (gsearch.best_params_, gsearch.best_score_)

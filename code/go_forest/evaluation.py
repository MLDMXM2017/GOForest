import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error, r2_score)
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, f1_score,
                             average_precision_score, precision_score)

metric_names_classification = [ "AUROC", 
                                "AUPR",
                                "F1",
                                "ACC",
                                'Precision',
                                'Recall',]
metric_names_regression = [ 'R2',
                            'PCC',
                            'Spearmanr',
                            'MSE',
                            'RMSE',
                            'MAE',
                            'MedAE',
                            'MAPE',]

def scores_clf(y, proba):
    """
    return
    ------
        report: dict
    """
    pred = np.argmax(proba, axis=1)
    score_list = [
        roc_auc_score(y, proba[:, 1]),
        average_precision_score(y, proba[:, 1]),
        f1_score(y, pred),
        accuracy_score(y ,pred),
        precision_score(y, pred),
        recall_score(y, pred),
    ]
    score_name = metric_names_classification
    score_df = pd.DataFrame(score_list, index=score_name)
    return score_df

def scores_rr(y_true, y_pred):
    score_list = [  r2(y_true, y_pred),
                    pearson_corr(y_true, y_pred),
                    spearman_corr(y_true, y_pred),
                    mse(y_true, y_pred),
                    rmse(y_true, y_pred),
                    mae(y_true, y_pred),
                    medae(y_true, y_pred),
                    mape(y_true, y_pred),  ]
    score_name = metric_names_regression
    score_df = pd.DataFrame(score_list, index=score_name)
    return score_df


def pearson_corr(x, y):  # 计算向量x和y的person相关性
    xx = x - np.mean(x)
    yy = y - np.mean(y)
    return np.sum(xx * yy) / (np.linalg.norm(xx) * np.linalg.norm(yy, 2))

def spearman_corr(x, y):
    return spearmanr(x, y)[0]

def r2(x, y):
    return r2_score(x, y)

def mae(x, y):
    return mean_absolute_error(x, y)

def mape(x, y):
    return mean_absolute_percentage_error(x, y)

def medae(x, y):
    return median_absolute_error(x, y)

def mse(x, y):
    return mean_squared_error(x, y)

def rmse(x, y):
    return np.sqrt(mse(x, y))

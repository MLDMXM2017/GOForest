import numpy as np
import pandas as pd

class RRWeighter:
    def __init__(
        self,
        # weight_type: str, 
        individual_names: list, 
        individual_scores: list,
        score_type: str = 'inverse',
        weight_strategy: str = '2',
        # scaler: float = 50, 
    ) -> None:
        """
        Parameters
        ----------
        individual_names: GOterm的名字
        """
        # self.weight_type: str = weight_type
        self.individual_names: tuple = tuple(individual_names)
        self.individual_score: list = individual_scores
        self.score_type: str = score_type
        self.weight_strategy: int = weight_strategy

        self._scaler_score()
    
    def _scaler_score(self):
        if self.score_type == "direct": # 如果score和准确率成正比, 比如r2, pcc等相关系数
            self.score = self.individual_score
        elif self.score_type == "inverse":  # 如果score和准确率成反比, 比如rmse, mae等loss
            pass

    def weight_strategy_1(self):
        weights = self.score / np.sum(self.score)
        return weights

    def weight_strategy_2(self, scaler=50):
        """
        Returns
        -------
        weights: (n_terms, )
        """
        score = self.individual_score
        mean_score = np.mean(score)
        difference = score - mean_score
        difference = difference*scaler  # 缩放差异
        exp_weights = np.exp(difference)
        weights = exp_weights/np.sum(exp_weights)
        return weights
    
    def weight_predict(self, y_pred_dict: dict, scaler=50):
        """加权预测
        parameters
        ----------
            y_pred_dict: dict
                预测结果的字典
                key: term
        """
        y_pred_list = self._processing_y_predict_dict(y_pred_dict)
        assert np.shape(y_pred_list)[1] == np.shape(self.individual_score)[0], \
            "the columns of y_pred_list must be same as the length of individual_score, this represented the number of terms"
        weighting_method = eval(f"self.weight_strategy_{self.weight_strategy}")
        weights = weighting_method(scaler)

        weighted_predict = np.sum(np.multiply(weights, y_pred_list), axis=1)
        self.weights = pd.DataFrame(weights, index=self.individual_names, columns=["weight"])
        return weighted_predict.reshape(-1)
    
    def _processing_y_predict_dict(self, y_pred_dict):
        """处理y_predict_dict字典
        
        Returns
        -------
        y_pred_list: ndarray, shape=(n_samples, n_terms)
        """
        from operator import itemgetter
        y_pred_list = itemgetter(*self.individual_names)(y_pred_dict)   # 抽取出的每一个元素都是二维的,因此y_pred_list是三维: (n_terms, n_samples, 1)
        return np.transpose(np.squeeze(y_pred_list))
        
 


class CLFWeighter:
    def __init__(
        self,
        # weight_type: str, 
        individual_names: list, 
        individual_scores: list,
        score_type: str = 'inverse',
        weight_strategy: str = '2',
    ) -> None:
        # self.weight_type: str = weight_type
        self.individual_names: tuple = tuple(individual_names)
        self.individual_score: list = individual_scores
        self.score_type: str = score_type
        self.weight_strategy: int = weight_strategy

        self._scaler_score()
    
    def _scaler_score(self):
        if self.score_type == "direct": # 如果score和准确率成正比, 比如r2, pcc等相关系数
            self.score = self.individual_score
        elif self.score_type == "inverse":  # 如果score和准确率成反比, 比如rmse, mae等loss
            pass

    def weight_strategy_1(self):
        weights = self.score / np.sum(self.score)
        return weights

    def weight_strategy_2(self, scaler=50):
        score = self.individual_score
        mean_score = np.mean(score)
        difference = score - mean_score
        difference = difference*scaler  # 缩放差异
        exp_weights = np.exp(difference)
        weights = exp_weights/np.sum(exp_weights)
        return weights
    
    def weight_predict(self, y_pred_dict: dict, scaler=50):
        """加权预测
        parameters
        ----------
            y_pred_dict: dict
                预测结果的字典
                key: term, value: 预测值列表(1-dim张量)
                """
        y_pred_list = self._processing_y_predict_dict(y_pred_dict)
        assert np.shape(y_pred_list)[1] == np.shape(self.individual_score)[0], \
            "the columns of y_pred_list must be same as the length of individual_score, this represented the number of terms"
        weighting_method = eval(f"self.weight_strategy_{self.weight_strategy}")
        weights = weighting_method(scaler)

        weighted_predict = np.sum(np.multiply(weights, y_pred_list), axis=1)
        self.weights = pd.DataFrame(weights, index=self.individual_names, columns=["weight"])
        return weighted_predict.reshape(-1)

    def weight_predict_proba(self, y_proba_dict: dict, scaler=50):
        """加权预测, 用于类别概率
        parameters
        ----------
            y_proba_dict: dict
                预测结果的字典
                key: term, value: 预测概率列表(2-dim张量)

        """
        y_proba_list = self._processing_y_predict_dict(y_proba_dict)
        assert np.shape(y_proba_list)[0] == np.shape(self.individual_score)[0], \
            "the columns of y_proba_list must be same as the length of individual_score, this represented the number of terms"
        weighting_method = eval(f"self.weight_strategy_{self.weight_strategy}")
        weights = weighting_method(scaler)
        weights = np.reshape(weights, (len(self.individual_score), 1, 1))
        weighted_proba = np.sum(np.multiply(weights, y_proba_list), axis=0)
        # 保存weights到self.weights
        self.weights = pd.DataFrame(np.squeeze(weights), index=self.individual_names, columns=["weight"])
        return np.squeeze(weighted_proba)

    def _processing_y_predict_dict(self, y_pred_dict):
        """抽取y_predict_dict的预测列表
        
        parameter
        ---------
            y_pred_dict: 预测值字典或者预测概率字典

        return
        ------
            如果是预测值字典: 返回的是(n_terms, n_samples, 1)的形状
            如果是预测概率字典: 返回的是(n_terms, n_samples, n_class)的形状
        """
        from operator import itemgetter
        y_pred_list = itemgetter(*self.individual_names)(y_pred_dict)   # 抽取出的每一个元素都是二维的,因此y_pred_list是三维: (n_terms, n_samples, 1)
        return y_pred_list

                   
    

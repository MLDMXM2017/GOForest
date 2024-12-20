from typing import Any, Sequence, Tuple, List, Union
import numpy as np
class Triple(list):
    """interface"""
    def __init__(self, values) -> None:
        super(Triple, self).__init__(values)
        pass
    def stringify(self) -> str:
        pass

class TripleSet:
    """list of Triple"""
    def __init__(self, triple_list) -> None:
        self.values = triple_list


class Antecedent(Triple):
    """三元组"""
    def __init__(self, values: Tuple):
        super().__init__(values)
        self.feature_index = self[0]
        self.operator: int = int(self[1])
        self.threshold: float = self[2]

    def stringify(self, feature_name=None) -> str:
        """将三元组转化成字符串, 用于输出"""
        if feature_name is None:
            feature_name = f"feature_{self.feature_index}"
        if self.operator == 0: 
            operator = "<"
        elif self.operator == 1:
            operator = ">=" 
        else:
            raise ValueError( "the operator must be 0 or 1 ! ")

        return f"{feature_name} {operator} {self.threshold}"

class Consequent(Triple):
    def __init__(self, values: Tuple):
        super().__init__(values)

    def stringify(self) -> str:
        return f"value = {self[-1]}"


class Rule:
    def __init__(
        self, 
        triples: Union[Sequence[Triple], TripleSet],
        feature_names: Sequence = None,
    ) -> None:
        """如果是空规则, 则应设置参数triples=[]"""
        try:
            if isinstance(triples, Sequence):
                self.triples: TripleSet = TripleSet(triples)
            elif isinstance(triples, TripleSet):
                self.triples: TripleSet = triples
            else:
                raise
            # self.antecedents: Sequence[Antecedent] = self.triples.values[0:-1]
            self.antecedents: Sequence[Antecedent] = self._sort_antecedent(self.triples.values[0:-1])
            self.consequent: Consequent = self.triples.values[-1]
            self.length: int = len(self.antecedents)
            if feature_names is not None:
                self.feaure_names: Sequence = feature_names
            else:
                self.feaure_names = [f"feature_{i}" for i in range(len(self.antecedents))]
            self.index_antecedent = np.array(self.antecedents, int)[:, 0]
            # print
        except IndexError: 
            self.length: int = 0
            self.feaure_names = None
            
            # print("The rule should not be empty. ")

    def _sort_antecedent(self, triples:Sequence[Triple]) -> Sequence[Antecedent]:
        """把规则前件进行排序"""
        dtype = [("index", int), ("symbol", int), ("threshold", float)]
        triples = np.array([tuple(item) for item in triples], dtype=dtype)
        triples_sorted = np.sort(triples, order="index")
        triples_sorted = [Antecedent(item) for item in triples_sorted]
        return triples_sorted
        

    def stringify(self, ) -> str:
        """将规则转化成if...else...形式的字符串"""
        if self.length == 0: return "None"

        string_rule = "if " 
        for antecedent in self.antecedents:
            feature_name = self.feaure_names[antecedent.feature_index]
            string_rule += antecedent.stringify(feature_name) + " and "
        string_rule += "then " + self.consequent.stringify()
        return string_rule
    def is_empty(self, ) -> bool:
        """判断本规则是否为空"""
        return True if self.length==0 else False

if __name__ == "__main__":
    a = Consequent(['value', 2, 500])
    a.value = 600
    print(a)


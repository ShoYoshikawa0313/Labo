import functools

@functools.total_ordering
class Mol_Data:
    def __init__(self, mol, smi: str, score, parent1_smi: str = "", parent2_smi: str = "", inter_smi: str = ""):
        self.mol = mol
        self.smi = smi
        self.score = score
        self.parent1_smi = parent1_smi
        self.parent2_smi = parent2_smi
        self.inter_smi = inter_smi
        
    # self < other の振る舞いを定義
    def __lt__(self, other):
        if not isinstance(other, Mol_Data):
            return NotImplemented
        # scoreで比較
        return self.score < other.score

    # self == other の振る舞いを定義
    def __eq__(self, other):
        if not isinstance(other, Mol_Data):
            return NotImplemented
        return self.score == other.score
    
    def to_dict(self):
        """
        オブジェクトのメンバ変数をJSON化可能な辞書形式で返す。
        (molオブジェクトは除外)
        """
        return {
            "smi": self.smi,
            "parent1_smi": self.parent1_smi,
            "parent2_smi": self.parent2_smi,
            "inter_smi": self.inter_smi,
            "score": self.score
        }
    
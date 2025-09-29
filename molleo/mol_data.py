import functools
from rdkit.Chem import Mol

# @functools.total_ordering を付けると、__lt__と__eq__を定義するだけで
# 他の比較演算子（<=, >, >=）も自動で実装してくれます。
@functools.total_ordering
class Mol_Data:
    def __init__(self, mol: Mol, smi: str, score: float, parent1_smi: str = "", parent2_smi: str = ""):
        self.mol = mol
        self.smi = smi
        self.score = score
        self.parent1_smi = parent1_smi
        self.parent2_smi = parent2_smi

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
        
    def __repr__(self):
        return f"Mol_Data(smi='{self.smi}', score={self.score})"
    
    def to_dict(self):
        """
        オブジェクトのメンバ変数をJSON化可能な辞書形式で返す。
        (molオブジェクトは除外)
        """
        return {
            "smi": self.smi,
            "score": self.score,
            "parent1_smi": self.parent1_smi,
            "parent2_smi": self.parent2_smi
        }
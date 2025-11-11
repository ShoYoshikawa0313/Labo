import functools
import math

from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import Mol, AllChem
from rdkit.DataStructs import TanimotoSimilarity

# @functools.total_ordering を付けると、__lt__と__eq__を定義するだけで
# 他の比較演算子（<=, >, >=）も自動で実装してくれます。
@functools.total_ordering
class Mol_Data:
    def __init__(self,task: str, mol: Mol, smi: str, parent1_smi: str = "", parent2_smi: str = "", inter_smi: str = ""):
        self.task = task
        self.mol = mol
        self.smi = smi
        self.parent1_smi = parent1_smi
        self.parent2_smi = parent2_smi
        self.inter_smi = inter_smi
        self._score = None

        self.task_evaluator = Oracle(name = self.task)

    def score_smi(self,smi):
        # SMILES文字列がNoneの場合、スコア0を返します。
        if smi is None:
            return 0
        # SMILES文字列からRDKitのMolオブジェクトを生成します。
        mol = Chem.MolFromSmiles(smi)
        # Molオブジェクトが生成できない、またはSMILES文字列が空の場合、スコア0を返します。
        if mol is None or len(smi) == 0:
            return 0
        else:
            # Molオブジェクトを正規化されたSMILES文字列に変換します。
            smi = Chem.MolToSmiles(mol)
            # 評価器を使用して分子のフィットネス（適合度）を計算します。
            fitness = float(self.task_evaluator(smi))
            # フィットネスがNaN（非数）の場合、0に設定します。
            if math.isnan(fitness):
                fitness = 0
            # オラクルの名前に "docking" が含まれている場合、フィットネスの符号を反転させます。
            if self.task == "docking":
                fitness *= -1

            # 分子バッファから該当するSMILES文字列のフィットネスを返します。
            return fitness
    
    @property
    def score(self):
        if self._score is not None:
            return self._score
        else:
            self._score = self.score_smi(self.smi)
            return self._score
        
    @property
    def similarities(self):
        sims = {}

        self_fp = AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=1024)

        if self.parent1_smi != "":
            fp_parent1 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(self.parent1_smi), 2, nBits=1024)
            sims["parent1"] = TanimotoSimilarity(fp_parent1,self_fp)
        if self.parent2_smi != "":
            fp_parent2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(self.parent2_smi), 2, nBits=1024)
            sims["parent2"] = TanimotoSimilarity(fp_parent2,self_fp)
        if self.inter_smi != "":
            fp_inter = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(self.inter_smi), 2, nBits=1024)
            sims["inter"] = TanimotoSimilarity(fp_inter,self_fp)

        return sims

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
            "task": self.task,
            "smi": self.smi,
            "parent1_smi": self.parent1_smi,
            "parent2_smi": self.parent2_smi,
            "inter_smi": self.inter_smi
        }
    
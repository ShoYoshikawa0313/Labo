import math
import tdc

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

class Evaluator:
    def __init__(self, task):
        self.task = task
        self.oracle = tdc.Oracle(name = self.task)
        self.diversity = tdc.Evaluator(name = 'Diversity')

    def __getstate__(self):
        # オブジェクトの状態をコピー
        state = self.__dict__.copy()
        # pickle化できない属性を削除
        del state['oracle']
        del state['diversity']
        return state

    def __setstate__(self, state):
        # オブジェクトの状態を復元
        self.__dict__.update(state)
        # pickle化できない属性を再初期化
        self.oracle = tdc.Oracle(name = self.task)
        self.diversity = tdc.Evaluator(name = 'Diversity')

    def score(self,smi):
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
            fitness = float(self.oracle(smi))
            # フィットネスがNaN（非数）の場合、0に設定します。
            if math.isnan(fitness):
                fitness = 0
            # オラクルの名前に "docking" が含まれている場合、フィットネスの符号を反転させます。
            if self.task == "docking":
                fitness *= -1

            # 分子バッファから該当するSMILES文字列のフィットネスを返します。
            return fitness
                

def calculate_population_similarity(population1, population2):
    """
    2つのpopulation間の類似度を「最大類似度の平均」を用いて計算します。
    具体的には、一方の集団の各個体について、もう一方の集団における最も類似度の高い個体との類似度（最大類似度）を求め、
    それらの平均値を取ります。これを双方向で行い、最終的に2つの平均値をさらに平均して返します。
    類似度計算にはMorganフィンガープリントとTanimoto係数を使用します。

    Args:
        population1 (list): Mol_Dataオブジェクトのリスト。
        population2 (list): Mol_Dataオブジェクトのリスト。

    Returns:
        float: 2つのpopulation間の平均最大類似度。
    """
    mols1 = [mol_data.mol for mol_data in population1 if mol_data.mol is not None]
    mols2 = [mol_data.mol for mol_data in population2 if mol_data.mol is not None]

    if not mols1 or not mols2:
        return 0.0

    fps1 = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols1]
    fps2 = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols2]

    def get_avg_max_similarity(fps_a, fps_b):
        total_max_similarity = 0
        if not fps_a or not fps_b:
            return 0.0
        
        for fp_a in fps_a:
            max_sim = 0
            for fp_b in fps_b:
                sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
                if sim > max_sim:
                    max_sim = sim
            total_max_similarity += max_sim
        return total_max_similarity / len(fps_a)

    avg_max_sim_1_to_2 = get_avg_max_similarity(fps1, fps2)
    avg_max_sim_2_to_1 = get_avg_max_similarity(fps2, fps1)

    return (avg_max_sim_1_to_2 + avg_max_sim_2_to_1) / 2.0
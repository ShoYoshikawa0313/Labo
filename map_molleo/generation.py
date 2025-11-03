import tqdm
import numpy as np  # 数値計算に使用

from rdkit import Chem, rdBase  # 分子操作のためのRDKitライブラリ
rdBase.DisableLog('rdApp.error')  # RDKitのエラーログを無効化

from tdc import Evaluator
from tdc.generation import MolGen # tdc.generation.MolGenクラスをインポートします。分子生成タスクのためのデータセットをロードします。

from mol_data import Mol_Data

# スコアが0になるのを防ぐための微小な値
MINIMUM = 1e-10

class Generation():
    '''
    遺伝的アルゴリズム（GA）をベースとした分子最適化を実行するクラス。
    '''

    def __init__(self, LLM, composit):

        self.LLM = LLM

        self.name = composit["name"]
        self.task = composit["task"]
        self.population_size = composit["population_size"]
        self.offspring_size = composit["offspring_size"]
        
        self.diversity_evaluator = Evaluator(name = 'Diversity') # 分子の多様性を評価するための評価器を初期化します。

        data = MolGen(name = 'ZINC') # ZINCデータセットをロードします。
        self.all_smiles = data.get_data()['smiles'].tolist() # データセットからSMILESのリストを取得します。
        self.all_smiles.sort()

        # 初期集団を決定 データセットからランダムに選択（探索）
        self.population = self.make_initial_population()
        self.scores = []
        self.diversities = []
        self.n_generation  = 0

    def make_initial_population(self):
        initial_smis = np.random.choice(self.all_smiles, self.population_size).tolist() 
        initial_population = []
        for smi in tqdm.tqdm(initial_smis):
            mol = Chem.MolFromSmiles(smi)
            initial_population.append(Mol_Data(self.task, mol, smi))
        initial_population.sort(reverse=True)
        return initial_population

    def weighted_random_select(self, size, reverse=False):
        # スコアを抽出
        population_scores = None
        if reverse == False:
            population_scores = [mlc.score for mlc in self.population]
        else:
            population_scores = [1.0 - mlc.score for mlc in self.population]
        # スコアと分子をタプルのリストにまとめる
        all_tuples = list(zip(population_scores, self.population))
        # スコアに微小な値を加えて、ゼロ除算を回避する
        population_scores = [s + MINIMUM for s in population_scores]
        # スコアの合計を計算
        sum_scores = sum(population_scores)
        # 各個体のスコアを正規化し、選択確率を計算
        population_probs = [p / sum_scores for p in population_scores]
        # 計算された確率分布に基づき、個体のインデックスを復元抽出で選択
        indices = np.random.choice(len(all_tuples), p=population_probs, size=size, replace=True)
        return indices

    def sanitize(self, mlcs): # 分子のリストをサニタイズ（検証・クリーンアップ）するメソッドです。
        new_mlcs = [] # 新しい分子のリストを初期化します。
        smi_set = set() # SMILES文字列のセットを初期化します（重複を避けるため）。
        for mlc in mlcs: # 各分子についてループします。
            smi = mlc.smi # 分子オブジェクトをSMILES文字列に変換します。
            if smi not in smi_set: # もしSMILESが有効で、まだセットになければ、
                smi_set.add(smi) # セットにSMILESを追加します。
                new_mlcs.append(mlc) # 新しいリストに分子オブジェクトを追加します。
        return new_mlcs # サニタイズされた分子のリストを返します。

    def generational_shift(self):
        next_population = self.population[:]

        # スコアに基づいて親集団（メイティングプール）を形成
        indices = self.weighted_random_select(self.offspring_size)
        mating_list = [next_population[index] for index in indices]

        offsprings = self.LLM.mating(mating_list)

        # 現世代の集団に新しく生成した子孫集団を追加
        next_population += offsprings
        # 無効な分子を除去（サニタイズ）
        next_population = self.sanitize(next_population)

        next_population = sorted(next_population, reverse=True)[:self.population_size]

        avg_score = np.mean([mlc.score for mlc in next_population])
        self.scores.append(avg_score)

        self.population = next_population
        self.n_generation += 1
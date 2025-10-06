import os
import random
import yaml
import math
import tqdm
from typing import List  # 型ヒントに使用
import numpy as np  # 数値計算に使用

from rdkit import Chem, rdBase  # 分子操作のためのRDKitライブラリ
rdBase.DisableLog('rdApp.error')  # RDKitのエラーログを無効化

from tdc import Oracle, Evaluator
from tdc.generation import MolGen # tdc.generation.MolGenクラスをインポートします。分子生成タスクのためのデータセットをロードします。

# 自作モジュールをインポート
from mol_data import Mol_Data
from molleo.OR_batch import OPEN_ROUTER
from biot5 import BioT5
from gemini import Gemini

# スコアが0になるのを防ぐための微小な値
MINIMUM = 1e-10

class GB_GA_Optimizer():
    '''
    遺伝的アルゴリズム（GA）をベースとした分子最適化を実行するクラス。
    '''

    def __init__(self, args=None):
        # 引数をインスタンス変数に保存
        self.args = args
        
        # 使用する分子言語モデル（MolLM）をインスタンス化
        if self.args.LLM == "GPT_OSS":
            self.LLM = OPEN_ROUTER(self.args, model="openai/gpt-oss-20b:free", interval=10)
        elif self.args.LLM == "Gemini":
            self.LLM = Gemini(self.args, model="gemini-2.5-flash", interval=10)
        elif self.args.LLM == "BioT5":
            self.LLM = BioT5(self.args)

        self.diversity_evaluator = Evaluator(name = 'Diversity') # 分子の多様性を評価するための評価器を初期化します。
        self.task_evaluator = Oracle(name = self.args.task)

        data = MolGen(name = 'ZINC') # ZINCデータセットをロードします。
        self.all_smiles = data.get_data()['smiles'].tolist() # データセットからSMILESのリストを取得します。
        self.all_smiles.sort()

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
            if "docking" in self.args.task:
                fitness *= -1

            # 分子バッファから該当するSMILES文字列のフィットネスを返します。
            return fitness

    def make_initial_population(self):
        initial_smis = np.random.choice(self.all_smiles, self.args.population_size).tolist() 
        initial_population = []
        for smi in tqdm.tqdm(initial_smis):
            mol = Chem.MolFromSmiles(smi)
            score = self.score_smi(smi)
            initial_population.append(Mol_Data(mol,smi,score))
        initial_population.sort(reverse=True)
        return initial_population


    def make_mating_pool(self, population: List[Mol_Data], offspring_size: int):
        '''
        Mol_Dataの集合が与えられ、
        scoresを重みとして復元抽出し、次世代の親となる個体群を作成します。
        Args:
            population (List[Mol_Data]): Mol_Dataオブジェクトのリスト。
            offspring_size (int): 生成する子孫の数。
        Returns:
            list: 選択されたMol_Dataオブジェクトのリスト（重複あり）。
        '''
        # スコアを抽出
        population_scores = [mlc.score for mlc in population]
        # スコアと分子をタプルのリストにまとめる
        all_tuples = list(zip(population_scores, population))
        # スコアに微小な値を加えて、ゼロ除算を回避する
        population_scores = [s + MINIMUM for s in population_scores]
        # スコアの合計を計算
        sum_scores = sum(population_scores)
        # 各個体のスコアを正規化し、選択確率を計算
        population_probs = [p / sum_scores for p in population_scores]
        # 計算された確率分布に基づき、親となる個体のインデックスを復元抽出で選択
        mating_indices = np.random.choice(len(all_tuples), p=population_probs, size=offspring_size, replace=True)
        # 選択されたインデックスに対応するMol_Dataオブジェクトを抽出
        mating_list = [all_tuples[indice][1] for indice in mating_indices]
        
        # メイティングリストを返す
        return mating_list

    def family2MolData(self, family):
        smi = family[0]
        mol = Chem.MolFromSmiles(smi)
        score = self.score_smi(smi)
        return Mol_Data(mol, smi, score, family[1], family[2])

    def sanitize(self, mlcs): # 分子のリストをサニタイズ（検証・クリーンアップ）するメソッドです。
        """
        Mol_Dataオブジェクトのリストを検証し、重複を除外します。
        """
        new_mlcs = [] # 新しい分子のリストを初期化します。
        smi_set = set() # SMILES文字列のセットを初期化します（重複を避けるため）。
        for mlc in mlcs: # 各分子についてループします。
            smi = mlc.smi # 分子オブジェクトをSMILES文字列に変換します。
            if smi not in smi_set: # もしSMILESが有効で、まだセットになければ、
                smi_set.add(smi) # セットにSMILESを追加します。
                new_mlcs.append(mlc) # 新しいリストに分子オブジェクトを追加します。
        return new_mlcs # サニタイズされた分子のリストを返します。
    
    def early_stop(self, scores):
        # スコアの履歴リストを受け取り、早期終了すべきかどうかを判断します。
        # 比較に必要なスコア数（patience + 1）が溜まっていない場合は、早期終了しません。
        if len(scores) < self.args.patience + 1:
            return False
        
        # スコアの向上が見られなかった回数をカウントするカウンター。
        cnt = 0
        # 直近のpatience回数分のスコアの変動をチェックします。
        for i in range(self.args.patience):
            # 最新のスコアと一つ前のスコアを比較します。
            new_score = scores[-(i+1)]
            old_score = scores[-(i+2)]
            # スコアの向上が閾値（1e-3）未満の場合、カウンターをインクリメントします。
            if(new_score - old_score) < 1e-3:
                cnt += 1
        
        # スコアが向上しなかった回数がpatience回数に達した場合、早期終了と判断します。
        if cnt >= self.args.patience:
            print("Early Stopping")
            return True
        
        # 早期終了の条件を満たさない場合はFalseを返します。
        return False

    def log_intermediate(self, n_generation, mlcs): # 中間結果をログに出力するメソッドです。
        """
        最適化プロセスの途中経過をコンソールに出力します。
        """
        len_mlcs = float(len(mlcs))
        sorted_mlcs = sorted(mlcs, reverse=True)
        scores = [mlc.score for mlc in sorted_mlcs]
        smis = [mlc.smi for mlc in sorted_mlcs]

        avg_top1 = np.max(scores[: math.ceil(len_mlcs*0.01)])
        avg_top10 = np.mean(scores[: math.ceil(len_mlcs*0.1)])
        avg_top50 = np.mean(scores[: math.ceil(len_mlcs*0.5)])
        avg_overall = np.mean(scores)
        diversity_overall = self.diversity_evaluator(smis)

        print(f'{n_generation}/{self.args.max_generations} | ' # 呼び出し回数と最大呼び出し回数を表示します。
                f'top1%: {avg_top1:.3f} | '  
                f'top10%: {avg_top10:.3f} | ' 
                f'top50%: {avg_top50:.3f} | ' 
                f'Overall: {avg_overall:.3f} | ' 
                f'div: {diversity_overall:.3f}  ' + 50*"-")

    def save_population(self, mlcs, suffix=None): # 結果を保存するメソッドです。
        """
        最適化によって得られた分子とそのスコアをYAMLファイルに保存します。
        """
        print(f"Saving population...") # "Saving molecules..."と表示します。
        output_file_path = os.path.join(self.args.output_dir, 'population_' + suffix + '.yaml') # 接尾辞を付けた出力ファイルパスを設定します。

        with open(output_file_path, 'w') as f: # 出力ファイルを書き込みモードで開きます。
            # SMILESをキー、スコアを値とする辞書を作成します。
            result_dict = { mlc.smi : mlc.to_dict() for mlc in mlcs}
            yaml.dump(result_dict, f, sort_keys=False) # 作成した辞書をYAML形式でファイルに書き込みます。

    def _optimize(self):
        '''
        最適化のメインループ。GAを実行して分子集団を世代ごとに進化させます。
        '''
        
        # 初期集団を決定 データセットからランダムに選択（探索）
        population = self.make_initial_population()
        # 早期終了判定のため、各世代の上位100位の平均スコアを格納するリストを初期化します。
        scores = []

        # GAのメインループを開始
        for n_generation in range(self.args.max_generations):

            self.log_intermediate(n_generation, population)
            self.save_population(population, f"{n_generation}G")

            # スコアに基づいて親集団（メイティングプール）を形成
            mating_list = self.make_mating_pool(population, self.args.population_size)
            
            # GPT-OSSを用いて分子を編集し、子孫を生成
            families = self.LLM.generational_shift(mating_list)
            offsprings = [self.family2MolData(family) for family in families]

            # 現世代の集団に新しく生成した子孫集団を追加
            population += offsprings
            # 無効な分子を除去（サニタイズ）
            population = self.sanitize(population)

            population = sorted(population, reverse=True)[:self.args.population_size]

            avg_score = np.mean([mlc.score for mlc in population])
            scores.append(avg_score)

            # --- 早期終了判定 ---
            if self.early_stop(scores):
                break
            
    def optimize(self): # 最適化のメインメソッドです。
        """
        分子最適化プロセス全体を実行します。
        乱数シードの設定、タスクのラベリング、最適化の実行、結果のロギングと保存、
        そして最後に状態のリセットを行います。
        """

        np.random.seed(self.args.seed) # numpyの乱数シードを設定します。
        random.seed(self.args.seed) # Pythonのrandomモジュールの乱数シードを設定します。

        self._optimize() # 内部の最適化メソッドを呼び出します。
import os
import random
import yaml
import math
from datetime import datetime
from typing import List  # 型ヒントに使用

import numpy as np  # 数値計算に使用

from rdkit import Chem, rdBase  # 分子操作のためのRDKitライブラリ
rdBase.DisableLog('rdApp.error')  # RDKitのエラーログを無効化

import tdc
from tdc.generation import MolGen # tdc.generation.MolGenクラスをインポートします。分子生成タスクのためのデータセットをロードします。

# 自作モジュールをインポート
from GPT_OSS import GPT_OSS

# スコアが0になるのを防ぐための微小な値
MINIMUM = 1e-10

def make_mating_pool(population_mol: List, population_scores, offspring_size: int):
    '''
    RDKit Molオブジェクトの集団とそのスコアを与えられ、
    population_scoresを重みとして復元抽出し、次世代の親となる個体群（メイティングプール）を作成します。
    Args:
        population_mol (List[Mol]): RDKit Molオブジェクトのリスト。
        population_scores (list): ScoringFunctionによって与えられた正規化されていないスコアのリスト。
        offspring_size (int): 生成する子孫の数。
    Returns:
        list: 選択された(スコア, Molオブジェクト)のタプルのリスト（重複あり）。
    '''
    # スコアと分子をタプルのリストにまとめる
    all_tuples = list(zip(population_scores, population_mol))
    # スコアに微小な値を加えて、ゼロ除算を回避する
    population_scores = [s + MINIMUM for s in population_scores]
    # スコアの合計を計算
    sum_scores = sum(population_scores)
    # 各個体のスコアを正規化し、選択確率を計算
    population_probs = [p / sum_scores for p in population_scores]
    # 計算された確率分布に基づき、親となる個体のインデックスを復元抽出で選択
    mating_indices = np.random.choice(len(all_tuples), p=population_probs, size=offspring_size, replace=True)
    
    # 選択されたインデックスに対応する(スコア, Molオブジェクト)のタプルを取得
    mating_tuples = [all_tuples[indice] for indice in mating_indices]
    
    # メイティングプールを返す
    return mating_tuples


class GB_GA_Optimizer():
    '''
    遺伝的アルゴリズム（GA）をベースとした分子最適化を実行するクラス。
    '''

    def __init__(self, args=None):
        # 引数をインスタンス変数に保存
        self.args = args
        
        # 使用する分子言語モデル（MolLM）をインスタンス化
        self.mol_lm = GPT_OSS()

        # taskを言語モデルに設定
        self.mol_lm.tasks = self.args.tasks
        self.task_evaluator = None
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity') # 分子の多様性を評価するための評価器を初期化します。

        data = MolGen(name = 'ZINC') # ZINCデータセットをロードします。
        self.all_smiles = data.get_data()['smiles'].tolist() # データセットからSMILESのリストを取得します。

    def assign_evaluator(self, evaluator): # 評価器を割り当てるメソッドです。
        self.task_evaluator = evaluator # 評価器をインスタンス変数に設定します。

    def population_sort(self, population_scores, population_mol, population_smiles):
        # スコアと分子をタプルにし、リスト化
        population_tuples = list(zip(population_scores, population_mol, population_smiles))
        # スコアに基づいて降順にソート
        population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:self.args.population_size]
        # タプルからMolオブジェクトとスコアを再度分離
        population_scores = [t[0] for t in population_tuples]
        population_mol = [t[1] for t in population_tuples]
        population_smiles = [t[2] for t in population_tuples]
        
        return population_scores, population_mol, population_smiles

    def sanitize(self, mol_list): # 分子のリストをサニタイズ（検証・クリーンアップ）するメソッドです。
        """
        RDKitのMolオブジェクトのリストを検証し、不正な分子や重複を除外します。

        Args:
            mol_list (list): RDKitのMolオブジェクトのリスト。

        Returns:
            list: サニタイズされたMolオブジェクトのリスト。
        """
        new_mol_list = [] # 新しい分子のリストを初期化します。
        smiles_set = set() # SMILES文字列のセットを初期化します（重複を避けるため）。
        for mol in mol_list: # 各分子についてループします。
            if mol is not None: # もし分子オブジェクトがNoneでなければ、
                try: # 例外処理を開始します。
                    smiles = Chem.MolToSmiles(mol) # 分子オブジェクトをSMILES文字列に変換します。
                    if smiles is not None and smiles not in smiles_set: # もしSMILESが有効で、まだセットになければ、
                        smiles_set.add(smiles) # セットにSMILESを追加します。
                        new_mol_list.append(mol) # 新しいリストに分子オブジェクトを追加します。
                except ValueError: # SMILESへの変換でエラーが発生した場合、
                    print('bad smiles') # 'bad smiles'と表示します。
        return new_mol_list # サニタイズされた分子のリストを返します。

    def log_intermediate(self, n_generation, smis, scores): # 中間結果をログに出力するメソッドです。
        """
        最適化プロセスの途中経過をコンソールに出力します。
        """
        temp_top100_smis = smis[:100] # バッファの上位100件を取得します。
        temp_top100_scores = scores[:100]

        avg_top1 = np.max(temp_top100_scores) # 上位1位のスコア（最大スコア）を計算します。
        avg_top10 = np.mean(sorted(temp_top100_scores, reverse=True)[:10]) # 上位10件のスコアの平均値を計算します。
        avg_top100 = np.mean(temp_top100_scores) # 上位100件のスコアの平均値を計算します。
        diversity_top100 = self.diversity_evaluator(temp_top100_smis) # 上位100件の分子の多様性を計算します。


        print(f'{n_generation}/{self.args.max_generations} | ' # 呼び出し回数と最大呼び出し回数を表示します。
                f'avg_top1: {avg_top1:.3f} | ' # 上位1位のスコアを表示します。
                f'avg_top10: {avg_top10:.3f} | ' # 上位10件の平均スコアを表示します。
                f'avg_top100: {avg_top100:.3f} | ' # 上位100件の平均スコアを表示します。
                f'div: {diversity_top100:.3f}') # 多様性を表示します。
        print()



    def save_result(self, smis, scores, suffix=None): # 結果を保存するメソッドです。
        """
        最適化によって得られた分子とそのスコアをYAMLファイルに保存します。

        Args:
            suffix (str, optional): 出力ファイル名に追加する接尾辞。 Defaults to None.
        """

        print(f"Saving molecules...") # "Saving molecules..."と表示します。

        date_str = datetime.now().strftime("%m-%d-%H-%M")
        output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + date_str + '.yaml') # 接尾辞を付けた出力ファイルパスを設定します。

        with open(output_file_path, 'w') as f: # 出力ファイルを書き込みモードで開きます。
            yaml.dump({'smiles': smis, 'scores': scores}, f, sort_keys=False) # バッファの内容をYAML形式でファイルに書き込みます。

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
    
    def score_smi(self, smi):
        """
        単一の分子（SMILES文字列）を評価し、そのスコアを計算します。

        Args:
            smi (str): 評価対象の分子を表すSMILES文字列。

        Returns:
            float: 分子の評価スコア。
        """

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
            if "docking" in self.args.tasks[0]:
                fitness *= -1

            # 分子バッファから該当するSMILES文字列のフィットネスを返します。
            return fitness
        
    def evaluate(self, smiles_lst): #複数の分子（SMILES文字列）を評価します。
        return [ self.score_smi(smi) for smi in smiles_lst]

    def _optimize(self, evaluator):
        '''
        最適化のメインループ。GAを実行して分子集団を世代ごとに進化させます。
        '''

        # 評価関数（オラクル）を設定
        self.assign_evaluator(evaluator)
        
        # 初期集団を決定 データセットからランダムに選択（探索）
        starting_population = np.random.choice(self.all_smiles, self.args.population_size)

        # 初期集団のSMILESをMolオブジェクトに変換し、スコアを計算
        population_smiles = starting_population
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_scores = self.evaluate([Chem.MolToSmiles(mol) for mol in population_mol])

        # 早期終了判定のため、各世代の上位100位の平均スコアを格納するリストを初期化します。
        scores = []

        population_scores, population_mol, population_smiles = self.population_sort(population_scores, population_mol, population_smiles)

        # GAのメインループを開始
        for n_generation in range(self.args.max_generations):

            if n_generation % self.args.freq_log == 0:
                self.log_intermediate(n_generation, population_smiles, population_scores)

            # スコアに基づいて親集団（メイティングプール）を形成
            mating_tuples = make_mating_pool(population_mol, population_scores, self.args.population_size)
            
            # GPT-OSSを用いて分子を編集し、子孫を生成
            offspring_mol = [self.mol_lm.edit(mating_tuples, self.args.mutation_rate) for _ in range(self.args.offspring_size)]

            # 現世代の集団に新しく生成した子孫集団を追加
            population_mol += offspring_mol
            # 無効な分子を除去（サニタイズ）
            population_mol = self.sanitize(population_mol)

            # 新しい世代のスコアを計算
            population_scores = self.evaluate([Chem.MolToSmiles(mol) for mol in population_mol])
            
            population_scores, population_mol, population_smiles = self.population_sort(population_scores, population_mol, population_smiles)

            avg_score = np.mean(population_scores)
            scores.append(avg_score)

            # --- 早期終了判定 ---
            if self.early_stop(scores):
                break

         # 結果を保存します。
        self.save_result(population_smiles, population_scores, self.args.mol_lm + "_")
            
    def optimize(self, evaluator, seed=0, project="test"): # 最適化のメインメソッドです。
        """
        分子最適化プロセス全体を実行します。
        乱数シードの設定、タスクのラベリング、最適化の実行、結果のロギングと保存、
        そして最後に状態のリセットを行います。

        Args:
            oracle (Oracle): スコアリングに使用するOracleオブジェクト。
            seed (int, optional): 乱数シード。 Defaults to 0.
            project (str, optional): プロジェクト名。 Defaults to "test".
        """

        np.random.seed(seed) # numpyの乱数シードを設定します。
        random.seed(seed) # Pythonのrandomモジュールの乱数シードを設定します。

        self._optimize(evaluator) # 内部の最適化メソッドを呼び出します。
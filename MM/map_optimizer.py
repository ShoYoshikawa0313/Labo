import os
import random
import yaml
import numpy as np  # 数値計算に使用
import math

from rdkit import rdBase  # 分子操作のためのRDKitライブラリ
rdBase.DisableLog('rdApp.error')  # RDKitのエラーログを無効化

from LLM_operator.biot5 import BioT5
from LLM_operator.llaSMol import LlaSMol
from LLM_operator.drug_assist import Drug_Assist
from LLM_operator.gemini import Gemini
from LLM_operator.openrouter import Open_Router
from LLM_operator.ollama import Ollama

from generation import Generation 

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

class Map_Optimizer:
    
    def __init__(self, args):
        self.args = args

        np.random.seed(self.args.seed) # numpyの乱数シードを設定します。
        random.seed(self.args.seed) # Pythonのrandomモジュールの乱数シードを設定します。

        self.islands = []

    def make_islands(self):
        composition = None
        with open(self.args.composition_file, 'r', encoding='utf-8') as f:
            composition = yaml.safe_load(f)[self.args.model]
        for comps in composition:
            LLM = None

            if comps["LLM"]["type"] == "clm":
                if comps["LLM"]["name"] == "BioT5":
                    LLM = BioT5(comps)
                elif comps["LLM"] == "LlaSMol":
                    LLM = LlaSMol(comps)
                elif comps["LLM"] == "DrugAssist":
                    LLM = Drug_Assist(comps)
            elif comps["LLM"]["type"] == "ollama":
                LLM = Ollama(comps)
            elif comps["LLM_type"] == "openrouter":
                LLM = Open_Router(comps)
            elif comps["LLM_type"] == "gemini":
                LLM = Gemini(comps)

            self.islands.append(Generation(LLM, comps))

    def select_immigration_source(self,target):
        while(True):
            source = random.randint(0,len(self.islands)-1)
            if target != source: return source

    def immigration(self):
        return self.args.immigration_rate > random.random()
    
    def departure(self, island):
        indices = island.weighted_random_select(self.args.immigrants_size)
        return [island.population[index] for index in indices]

    def entry(self, island, immigrants):
        indices = island.weighted_random_select(self.args.immigrants_size, reverse=True)
        for i, immigrant in enumerate(immigrants):
            island.population[indices[i]] = immigrant
    
    def island_early_stop(self,island):
        # スコアの履歴リストを受け取り、早期終了すべきかどうかを判断します。
        # 比較に必要なスコア数（patience + 1）が溜まっていない場合は、早期終了しません。
        if len(island.scores) < self.args.patience + 1:
            return False
        
        # スコアの向上が見られなかった回数をカウントするカウンター。
        cnt = 0
        # 直近のpatience回数分のスコアの変動をチェックします。
        for i in range(self.args.patience):
            # 最新のスコアと一つ前のスコアを比較します。
            new_score = island.scores[-(i+1)]
            old_score = island.scores[-(i+2)]
            # スコアの向上が閾値（1e-3）未満の場合、カウンターをインクリメントします。
            if(new_score - old_score) < 1e-3:
                cnt += 1
        
        # スコアが向上しなかった回数がpatience回数に達した場合、早期終了と判断します。
        if cnt >= self.args.patience:
            print("Early Stopping")
            return True
        
        # 早期終了の条件を満たさない場合はFalseを返します。
        return False

    def log_intermediate(self, island): # 中間結果をログに出力するメソッドです。
        """
        最適化プロセスの途中経過をコンソールに出力します。
        """
        len_mlcs = float(len(island.population))
        sorted_mlcs = sorted(island.population, reverse=True)
        scores = [mlc.score for mlc in sorted_mlcs]
        smis = [mlc.smi for mlc in sorted_mlcs]

        avg_top1 = np.max(scores[: math.ceil(len_mlcs*0.01)])
        avg_top10 = np.mean(scores[: math.ceil(len_mlcs*0.1)])
        avg_top50 = np.mean(scores[: math.ceil(len_mlcs*0.5)])
        avg_overall = np.mean(scores)
        diversity_overall = island.diversity_evaluator(smis)
        
        print(f'{island.name} {island.n_generation}/{self.args.max_generations} | ' # 呼び出し回数と最大呼び出し回数を表示します。
                f'top1%: {avg_top1:.3f} | '  
                f'top10%: {avg_top10:.3f} | ' 
                f'top50%: {avg_top50:.3f} | ' 
                f'Overall: {avg_overall:.3f} | ' 
                f'div: {diversity_overall:.3f}  ' + 50*"-")

    def save_population(self, island, suffix=None): # 結果を保存するメソッドです。
        """
        最適化によって得られた分子とそのスコアをYAMLファイルに保存します。
        """
        print(f"Saving population...") # "Saving molecules..."と表示します。

        output_dir = os.path.join(self.args.root_output_dir, island.name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        population_dir = os.path.join(output_dir, "population")
        if not os.path.exists(population_dir):
            os.mkdir(population_dir)
        output_file_path = os.path.join(population_dir, 'population_' + suffix + '.yaml') # 接尾辞を付けた出力ファイルパスを設定します。

        with open(output_file_path, 'w') as f: # 出力ファイルを書き込みモードで開きます。
            # SMILESをキー、スコアを値とする辞書を作成します。
            result_dict = { mlc.smi : mlc.to_dict() for mlc in island.population}
            yaml.dump(result_dict, f, sort_keys=False) # 作成した辞書をYAML形式でファイルに書き込みます。

    def save_offspring(self, island, suffix=None): # 結果を保存するメソッドです。
        """
        最適化によって得られた分子とそのスコアをYAMLファイルに保存します。
        """
        print(f"Saving Child...") # "Saving molecules..."と表示します。

        output_dir = os.path.join(self.args.root_output_dir, island.name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        offspring_dir = os.path.join(output_dir, "offspring")
        if not os.path.exists(offspring_dir):
            os.mkdir(offspring_dir)
        output_file_path = os.path.join(offspring_dir, 'offspring_' + suffix + '.yaml') # 接尾辞を付けた出力ファイルパスを設定します。

        with open(output_file_path, 'w') as f: # 出力ファイルを書き込みモードで開きます。
            # SMILESをキー、スコアを値とする辞書を作成します。
            result_dict = { mlc.smi : mlc.to_dict() for mlc in island.offspring}
            yaml.dump(result_dict, f, sort_keys=False) # 作成した辞書をYAML形式でファイルに書き込みます。
    
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

    def finish(self):
        cnt = 0
        for island in self.islands:
            if self.island_early_stop(island): cnt += 1
        return cnt == len(self.islands)
    
    def optimize(self):
        while(True):
            for idx, island in enumerate(self.islands):

                self.log_intermediate(island)
                self.save_population(island,f"{island.n_generation}G")
                self.save_offspring(island,f"{island.n_generation}G")

                if self.island_early_stop(island): continue

                if self.immigration():
                    immigration_source = self.select_immigration_source(idx)
                    immigrants = self.departure(island)
                    source_name = self.islands[immigration_source].name
                    target_name = island.name
                    print(f"immigration : island{source_name} -> island{target_name}")
                    self.entry(island,immigrants)

                island.generational_shift()

            if self.finish(): break

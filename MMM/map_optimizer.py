import os
import random
import yaml
import numpy as np  # 数値計算に使用
import math
import multiprocessing

from rdkit import rdBase  # 分子操作のためのRDKitライブラリ
rdBase.DisableLog('rdApp.error')  # RDKitのエラーログを無効化

from LLM_operator.biot5 import BioT5
from LLM_operator.llaSMol import LlaSMol
from LLM_operator.drug_assist import Drug_Assist
from LLM_operator.gemini import Gemini
from LLM_operator.openrouter import Open_Router
from LLM_operator.ollama import Ollama

from island import Island
from evaluator import Evaluator

def _optimize_process(optimizer_and_island):
    optimizer, island = optimizer_and_island
    while not island.early_stop(optimizer.args.patience):
        optimizer.log_intermediate(island)
        optimizer.save_population(island, f"{island.n_generation}G")
        optimizer.save_offspring(island, f"{island.n_generation}G")
        island.generational_shift()
    return

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
                elif comps["LLM"]["name"] == "LlaSMol":
                    LLM = LlaSMol(comps)
                elif comps["LLM"]["name"] == "DrugAssist":
                    LLM = Drug_Assist(comps)
            elif comps["LLM"]["type"] == "ollama":
                LLM = Ollama(comps)
            elif comps["LLM"]["name"] == "openrouter":
                LLM = Open_Router(comps)
            elif comps["LLM"]["name"] == "gemini":
                LLM = Gemini(comps)

            self.islands.append(Island(LLM, Evaluator(comps["task"]), comps))

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
        diversity_overall = island.evaluator.diversity(smis)
        
        print(f' ID{os.getpid()} {island.name} {island.n_generation}/{self.args.max_generations} | ' # 呼び出し回数と最大呼び出し回数を表示します。
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
    
    def finish(self):
        cnt = 0
        for island in self.islands:
            if island.early_stop(self.args.patience): cnt += 1
        return cnt == len(self.islands)
    
    def optimize(self):
        num_processes = 4
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(_optimize_process, [(self, island) for island in self.islands])

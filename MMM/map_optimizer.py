import random
import yaml
import numpy as np  # 数値計算に使用
import multiprocessing

from rdkit import rdBase  # 分子操作のためのRDKitライブラリ
rdBase.DisableLog('rdApp.error')  # RDKitのエラーログを無効化

from LLM_operator.biot5 import BioT5
from LLM_operator.llaSMol import LlaSMol
from LLM_operator.drug_assist import Drug_Assist
from LLM_operator.gemini import Gemini
from LLM_operator.ollama import Ollama

from island import Island
from evaluator import Evaluator

def parallel_shift(island, trials):
    for _ in range(trials):
        island.generational_shift()
    return island

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
            elif comps["LLM"]["type"] == "gemini":
                LLM = Gemini(comps)

            self.islands.append(Island(LLM, Evaluator(comps["task"]), self.args.root_output_dir, comps))

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

    def finish(self):
        cnt = 0
        for island in self.islands:
            if island.early_stop(self.args.patience): cnt += 1
        return cnt == len(self.islands)
    
    def optimize(self):
        print(f"Max processes : {multiprocessing.cpu_count()}")
        num_processes = 1
        trials = 1
        while(self.finish() == False):
            with multiprocessing.Pool(processes=num_processes) as pool:
                self.islands = pool.starmap(parallel_shift,[(island,trials) for island in self.islands] )

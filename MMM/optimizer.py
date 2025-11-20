import random
import yaml
import numpy as np  # 数値計算に使用
import os
from joblib import Parallel, delayed
import logging

# tdcライブラリからのINFOレベルのログ(Found local copy...など)を抑制
logging.getLogger('tdc').setLevel(logging.WARNING)

from rdkit import rdBase  # 分子操作のためのRDKitライブラリ
rdBase.DisableLog('rdApp.error')  # RDKitのエラーログを無効化

from LLM_operator.biot5 import BioT5
from LLM_operator.llaSMol import LlaSMol
from LLM_operator.drug_assist import Drug_Assist
from LLM_operator.gemini import Gemini
from LLM_operator.ollama import Ollama

from island import Island
from evaluator import Evaluator

def parallel_shift(island, trials, process_id):
    for i in range(trials):
        island.generational_shift(process_id)
    return island

class Random_Optimizer:
    
    def __init__(self, args):
        self.args = args

        np.random.seed(self.args.seed) # numpyの乱数シードを設定します。
        random.seed(self.args.seed) # Pythonのrandomモジュールの乱数シードを設定します。

        composition = self.load_composition()

        self.max_generations = composition["settings"]["max_generations"]
        self.patience = composition["settings"]["patience"]
        self.immigrants_size = composition["settings"]["immigration_size"]

        self.islands = []
        self.make_islands(composition)

    def load_composition(self):
        composition = None
        with open(self.args.composition_file, 'r', encoding='utf-8') as f:
            composition = yaml.safe_load(f)[self.args.model]
        return composition

    def make_islands(self, composition):
        for comps in composition["operetors"]:
            LLM = None
            if comps["LLM"]["type"] == "clm":
                if comps["LLM"]["name"] == "BioT5":
                    LLM = BioT5(comps)
                elif comps["LLM"]["name"] == "LlaSMol":
                    LLM = LlaSMol(comps)
                elif comps["LLM"]["name"] == "drugassist-instruct":
                    LLM = Drug_Assist(comps)
                else: return -1
            elif comps["LLM"]["type"] == "ollama":
                LLM = Ollama(comps)
            elif comps["LLM"]["type"] == "gemini":
                LLM = Gemini(comps)
            else:
                return -1
            self.islands.append(Island(LLM, Evaluator(comps["task"]), self.args.root_output_dir, comps))

    def immigration_process(self):
        # 各島から移住させる個体（移民）を格納するリスト
        immigrats_islands = []
        if len(self.islands) <= 1:
            return
        # 各島（宛先）に対して、別の島（供給源）から移民を受け入れるプロセス
        for i in range(len(self.islands)):
            source = None
            # 宛先の島と供給源の島が同じにならないように、ランダムに供給源の島を選択する
            while(True):
                candidate = random.randint(0,len(self.islands)-1)
                if i != candidate:
                    source = candidate
                    break
            
            # 供給源の島から、評価値に基づいて重み付けランダムサンプリングで移民を選択する
            indices = self.islands[source].weighted_random_select(self.immigrants_size)
            # 選択された移民の個体をリストに追加する
            immigrats_islands.append([self.islands[source].population[index] for index in indices])
        
        # 各島で、評価の低い個体を移民と入れ替えるプロセス
        for i in range(len(self.islands)):
            # 宛先の島から、評価値が低い個体を逆重み付けランダムサンプリングで選択する
            # これにより、評価の低い個体が置換の対象となる
            indices = self.islands[i].weighted_random_select(self.immigrants_size, reverse=True)
            # 選択された評価の低い個体を、対応する移民の個体と入れ替える
            for j , index in enumerate(indices):
                self.islands[i].population[index] = immigrats_islands[i][j]
        

    def finish(self):
        cnt = 0
        for island in self.islands:
            if island.early_stop(self.patience) or island.n_generation >= self.max_generations:
                cnt += 1
        return cnt == len(self.islands)
    
    def optimize(self):
        print(f"Max processes : {os.cpu_count()}")
        while(self.finish() == False):
            self.islands = Parallel(n_jobs=self.args.processes)(
                delayed(parallel_shift)(island, self.args.immigration_freq, i) for i, island in enumerate(self.islands)
            )

            for island in self.islands:
                island.log_intermediate()

            self.immigration_process()

import random
import yaml
import numpy as np  # 数値計算に使用
import os
from joblib import Parallel, delayed
import glob

from rdkit import rdBase  # 分子操作のためのRDKitライブラリ
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
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

class Optimizer:
    
    def __init__(self, args):
        self.args = args

        np.random.seed(self.args.seed) # numpyの乱数シードを設定します。
        random.seed(self.args.seed) # Pythonのrandomモジュールの乱数シードを設定します。

        composition = self.load_composition()

        self.max_generations = composition["settings"]["max_generations"]
        self.immigrants_size = composition["settings"]["immigration_size"]
        self.processes = composition["settings"]["processes"]
        self.immigration_frequency = composition["settings"]["immigration_frequency"]
        self.immigration_type = composition["settings"]["immigration_type"]

        self.islands = []
        if self.args.resume == "": self.make_islands(composition)
        elif self.args.resume != "": self.resume(composition)

    def load_composition(self):
        with open(self.args.composition_file, 'r', encoding='utf-8') as f:
            composition = yaml.safe_load(f)[self.args.model]
        return composition

    def make_islands(self, composition):
        for comps in composition["operetors"]:
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

    def resume(self, composition):
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

            resume_root = os.path.join(self.args.resume,comps["name"])
            population_path = os.path.join(resume_root,"population")
            offspring_path = os.path.join(resume_root,"offspring")

            population_files = glob.glob(os.path.join(population_path, 'population_*.yaml'))
            offspring_files = glob.glob(os.path.join(offspring_path, 'offspring_*.yaml'))

            if len(population_files) != len(offspring_files):
                return -1
            
            tmp = Island(LLM, Evaluator(comps["task"]), self.args.root_output_dir, comps, os.path.join(population_path,f"population_{len(population_files) -1}G.yaml"))

            self.islands.append(tmp)

    def random_immigration(self):
        # 各島から移住させる個体（移民）を格納するリスト
        islands_immigrants = [[] for _ in self.islands]
        
        if len(self.islands) <= 1:
            return
        # 各島（宛先）に対して、別の島（供給源）から移民を受け入れるプロセス
        for target in range(len(self.islands)):
            # 宛先の島と供給源の島が同じにならないように、ランダムに供給源の島を選択する
            while(True):
                source = random.randint(0,len(self.islands)-1)
                if target != source:
                    break
            
            target_smis = [mlc.smi for mlc in self.islands[target].population]
            while(len(islands_immigrants[target]) < self.immigrants_size):
                index = self.islands[source].weighted_random_select(1)[0]
                immigrant_mlc = self.islands[source].population[index]
                if immigrant_mlc.smi not in target_smis:
                    islands_immigrants[target].append(immigrant_mlc)

        # 各島で、評価の低い個体を移民と入れ替えるプロセス
        for target_index in range(len(self.islands)):
            # 宛先の島から、評価値が低い個体を逆重み付けランダムサンプリングで選択する
            # これにより、評価の低い個体が置換の対象となる
            indices = self.islands[target_index].weighted_random_select(self.immigrants_size, reverse=True)
            # 選択された評価の低い個体を、対応する移民の個体と入れ替える
            for j, index in enumerate(indices):
                self.islands[target_index].population[index] = islands_immigrants[target_index][j]
            self.islands[target_index].population.sort(reverse=True)

    def cluster_population(self, population, cutoff=0.2):
        """
        Perform Butina clustering on a population based on molecular fingerprints.

        Args:
            population (list): A list of Mol_Data objects.
            cutoff (float): Tanimoto similarity cutoff for clustering.

        Returns:
            list: A list of clusters, where each cluster is a list of Mol_Data objects.
        """
        # Generate fingerprints for all molecules in the population
        fps = [AllChem.GetMorganFingerprintAsBitVect(mlc.mol, radius=2) for mlc in population]

        # Calculate pairwise Tanimoto similarities
        n = len(fps)
        dists = []
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - sim for sim in sims])  # Convert similarity to distance

        # Perform Butina clustering
        clusters = Butina.ClusterData(dists, nPts=n, distThresh=cutoff, isDistData=True)

        # Map cluster indices to Mol_Data objects
        clustered_population = [[population[idx] for idx in cluster] for cluster in clusters]

        return clustered_population

    def max_similarity(self, mlc, population):
        max_sim = 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mlc.mol, radius=2)
        for other_mlc in population:
            fp2 = AllChem.GetMorganFingerprintAsBitVect(other_mlc.mol, radius=2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            if sim > max_sim:
                max_sim = sim
        return max_sim

    def cluster_novelty_immigration(self, max_cluster_size=5, cut_off_threshold=0.3): 
        if len(self.islands) <= 1:
            return
        
        islands_immigrants = [[] for _ in self.islands]
        islands_removes = [[] for _ in self.islands]
        
        # 各島をループして、クラスタの過密抑制とニッチ充填を行う
        for target_index in range(len(self.islands)):
            target_island = self.islands[target_index]
            
            # 1. 過密クラスタから個体を削除する
            # 島内の個体群をクラスタリング
            clusters = self.cluster_population(self.islands[target_index].population, cutoff=cut_off_threshold)
            
            remove_mlcs_index = []
            for cluster in clusters:
                # クラスタサイズが上限を超えている場合
                if len(cluster) > max_cluster_size:
                    # スコアでソートし、スコアの低い個体（超過分）を削除対象とする
                    sorted_cluster = sorted(cluster, reverse=True)
                    remove_mlcs = sorted_cluster[max_cluster_size:]
                    remove_mlcs_index.extend([target_island.population.index(mlc) for mlc in remove_mlcs if target_island.population.index(mlc) not in remove_mlcs_index])

            immigrants = []
            # 2. 削除して空いたスペースに、島に存在しない有望なクラスタから個体を補充する
            if len(remove_mlcs_index) > 0:
                target_smis = {mlc.smi for mlc in target_island.population}

                other_population = []
                for idx in range(len(self.islands)):
                    if idx == target_index:
                        continue
                    other_population.extend(self.islands[idx].population)

                # 他の島の個体群から、現在の島に対する新規性が高い（最大類似度が低い）個体を選ぶ
                # (個体, 現在の島との最大類似度) のタプルのリストを作成
                immigrant_candidates = [(mlc, self.max_similarity(mlc, target_island.population)) for mlc in other_population if mlc.smi not in target_smis]
                # 最大類似度が低い順（新規性が高い順）にソート
                immigrant_candidates.sort(key=lambda x: x[1])
                # 移住させる個体を決定
                immigrants = [mlc for mlc, _ in immigrant_candidates]
                immigrants = immigrants[:len(remove_mlcs_index)]

                print(f"Island {target_index}: Removed {len(remove_mlcs_index)} individuals, Immigrated {len(immigrants)} individuals.")
            
            islands_removes[target_index].extend(remove_mlcs_index) 
            islands_immigrants[target_index].extend(immigrants)

        # 各島で、評価の低い個体を移民と入れ替えるプロセス
        for target_index in range(len(self.islands)):
            for j, remove_index in enumerate(islands_removes[target_index]):
                self.islands[target_index].population[remove_index] = islands_immigrants[target_index][j]
            self.islands[target_index].population.sort(reverse=True)

    def finish(self):
        cnt = 0
        for island in self.islands:
            if island.n_generation >= self.max_generations:
                cnt += 1
        return cnt == len(self.islands)
    
    def optimize(self):
        print(f"Max processes : {os.cpu_count()}")
        while(self.finish() == False):
            self.islands = Parallel(n_jobs=self.processes)(
                delayed(parallel_shift)(island, self.immigration_frequency, i) for i, island in enumerate(self.islands)
            )

            print("Before Immigration:")
            for island in self.islands:
                island.log_intermediate()

            if self.immigration_type == "no_immigration":
                pass
            elif self.immigration_type == "random":
                self.random_immigration()
            elif self.immigration_type == "cluster_novelty":
                self.cluster_novelty_immigration(max_cluster_size=3,cut_off_threshold=0.5)
            else:
                pass
            
            print("After Immigration:")
            for island in self.islands:
                island.log_intermediate()

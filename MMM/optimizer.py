import random
import yaml
import numpy as np  # 数値計算に使用
import os
from joblib import Parallel, delayed

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
from evaluator import population_similarity
from rdkit.Chem.Scaffolds import MurckoScaffold

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

    def diversity_immigration(self):
        # 各島から移住させる個体（移民）を格納するリスト
        immigrats_islands = [[] for _ in self.islands]
        
        if len(self.islands) <= 1:
            return
        # 各島（宛先）に対して、別の島（供給源）から移民を受け入れるプロセス
        for target in range(len(self.islands)):
            
            min_similarity = float('inf')
            source = None
            # 最も類似度の低い島を移住元として選択
            for candidate in range(len(self.islands)):
                if target == candidate:
                    continue
                
                similarity = population_similarity(self.islands[target].population, self.islands[candidate].population)
                if similarity < min_similarity:
                    min_similarity = similarity
                    source = candidate
            
            target_smis = [mlc.smi for mlc in self.islands[target].population]
            while(len(immigrats_islands[target]) < self.immigrants_size):
                index = self.islands[source].weighted_random_select(1)[0]
                immigrant_mlc = self.islands[source].population[index]
                if immigrant_mlc.smi not in target_smis:
                    immigrats_islands[target].append(immigrant_mlc)

        # 各島で、評価の低い個体を移民と入れ替えるプロセス
        for i in range(len(self.islands)):
            # 宛先の島から、評価値が低い個体を逆重み付けランダムサンプリングで選択する
            # これにより、評価の低い個体が置換の対象となる
            indices = self.islands[i].weighted_random_select(self.immigrants_size, reverse=True)
            # 選択された評価の低い個体を、対応する移民の個体と入れ替える
            for j, index in enumerate(indices):
                self.islands[i].population[index] = immigrats_islands[i][j]

    def score_immigration(self):
        # 各島から移住させる個体（移民）を格納するリスト
        immigrats_islands = [[] for _ in self.islands]
        
        if len(self.islands) <= 1:
            return
        # 各島（宛先）に対して、別の島（供給源）から移民を受け入れるプロセス
        for target in range(len(self.islands)):
            
            max_score = 0.0
            source = None
            # 最もスコアの高い島を移住元として選択
            for candidate in range(len(self.islands)):
                if target == candidate:
                    continue
                
                score = self.islands[candidate].scores[-1]
                if score > max_score:
                    max_score = score
                    source = candidate
            
            target_smis = [mlc.smi for mlc in self.islands[target].population]
            while(len(immigrats_islands[target]) < self.immigrants_size):
                index = self.islands[source].weighted_random_select(1)[0]
                immigrant_mlc = self.islands[source].population[index]
                if immigrant_mlc.smi not in target_smis:
                    immigrats_islands[target].append(immigrant_mlc)

        # 各島で、評価の低い個体を移民と入れ替えるプロセス
        for i in range(len(self.islands)):
            # 宛先の島から、評価値が低い個体を逆重み付けランダムサンプリングで選択する
            # これにより、評価の低い個体が置換の対象となる
            indices = self.islands[i].weighted_random_select(self.immigrants_size, reverse=True)
            # 選択された評価の低い個体を、対応する移民の個体と入れ替える
            for j, index in enumerate(indices):
                self.islands[i].population[index] = immigrats_islands[i][j]

    def random_immigration(self):
        # 各島から移住させる個体（移民）を格納するリスト
        immigrats_islands = [[] for _ in self.islands]
        
        if len(self.islands) <= 1:
            return
        # 各島（宛先）に対して、別の島（供給源）から移民を受け入れるプロセス
        for target in range(len(self.islands)):
            source = None
            # 宛先の島と供給源の島が同じにならないように、ランダムに供給源の島を選択する
            while(True):
                candidate = random.randint(0,len(self.islands)-1)
                if target != candidate:
                    source = candidate
                    break
            
            target_smis = [mlc.smi for mlc in self.islands[target].population]
            while(len(immigrats_islands[target]) < self.immigrants_size):
                index = self.islands[source].weighted_random_select(1)[0]
                immigrant_mlc = self.islands[source].population[index]
                if immigrant_mlc.smi not in target_smis:
                    immigrats_islands[target].append(immigrant_mlc)

        # 各島で、評価の低い個体を移民と入れ替えるプロセス
        for i in range(len(self.islands)):
            # 宛先の島から、評価値が低い個体を逆重み付けランダムサンプリングで選択する
            # これにより、評価の低い個体が置換の対象となる
            indices = self.islands[i].weighted_random_select(self.immigrants_size, reverse=True)
            # 選択された評価の低い個体を、対応する移民の個体と入れ替える
            for j, index in enumerate(indices):
                self.islands[i].population[index] = immigrats_islands[i][j]

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

    def cluster_niche_filling_immigration(self, max_cluster_size=5, cut_off_threshold=0.3): 
        if len(self.islands) <= 1:
            return
        
        all_smis = set()
        all_mlcs = []
        for island in self.islands:
            for mlc in island.population:
                if mlc.smi not in all_smis:
                    all_mlcs.append(mlc)
                    all_smis.add(mlc.smi)
        
        # 各島をループして、クラスタの過密抑制とニッチ充填を行う
        for target_index in range(len(self.islands)):
            target_island = self.islands[target_index]
            
            # 1. 過密クラスタから個体を削除する
            # 島内の個体群をクラスタリング
            clusters = self.cluster_population(self.islands[target_index].population, cutoff=cut_off_threshold)
            
            removed_count = 0
            for cluster in clusters:
                # クラスタサイズが上限を超えている場合
                if len(cluster) > max_cluster_size:
                    # スコアでソートし、スコアの低い個体（超過分）を削除対象とする
                    sorted_cluster = sorted(cluster, reverse=True)
                    remove_mlcs = sorted_cluster[max_cluster_size:]
                    for mlc in remove_mlcs:
                        # population.remove()は低速なため、try-exceptで安全に実行
                        try:
                            target_island.population.remove(mlc)
                            removed_count += 1
                        except ValueError:
                            # 複数のクラスタに同じ個体が含まれる場合など、すでに削除されている可能性がある
                            pass

            # 2. 削除して空いたスペースに、島に存在しない有望なクラスタから個体を補充する
            if removed_count > 0:
                target_smis = {mlc.smi for mlc in target_island.population}
                immigrants = []

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
                immigrants = immigrants[:removed_count]

                print(f"Island {target_index}: Removed {removed_count} individuals, Immigrated {len(immigrants)} individuals.")
                target_island.population.extend(immigrants)

    def cluster_filling_diversity_immigration(self, max_cluster_size=5, cut_off_threshold=0.3): 
        if len(self.islands) <= 1:
            return
        
        all_smis = set()
        all_mlcs = []
        for island in self.islands:
            for mlc in island.population:
                if mlc.smi not in all_smis:
                    all_mlcs.append(mlc)
                    all_smis.add(mlc.smi)
        
        all_clusters = self.cluster_population(all_mlcs, cutoff=cut_off_threshold)
        # 各クラスターの平均スコアで降順にソート
        all_clusters.sort(key=lambda c: sum(mlc.score for mlc in c) / len(c) if c else 0, reverse=True)

        # 各島をループして、クラスタの過密抑制とニッチ充填を行う
        for target_index in range(len(self.islands)):
            target_island = self.islands[target_index]
            
            # 1. 過密クラスタから個体を削除する
            # 島内の個体群をクラスタリング
            clusters = self.cluster_population(self.islands[target_index].population, cutoff=cut_off_threshold)
            
            removed_count = 0
            for cluster in clusters:
                # クラスタサイズが上限を超えている場合
                if len(cluster) > max_cluster_size:
                    # スコアでソートし、スコアの低い個体（超過分）を削除対象とする
                    sorted_cluster = sorted(cluster, reverse=True)
                    remove_mlcs = sorted_cluster[max_cluster_size:]
                    for mlc in remove_mlcs:
                        # population.remove()は低速なため、try-exceptで安全に実行
                        try:
                            target_island.population.remove(mlc)
                            removed_count += 1
                        except ValueError:
                            # 複数のクラスタに同じ個体が含まれる場合など、すでに削除されている可能性がある
                            pass

            # 2. 削除して空いたスペースに、島に存在しない有望なクラスタから個体を補充する
            if removed_count > 0:
                # 島に存在するSMILESをセットに格納し、重複チェックを高速化
                target_smis = {mlc.smi for mlc in target_island.population}
                immigrants = []
                
                source_index = None
                min_similarity = float('inf')

                for candidate_index in range(len(self.islands)):
                    if target_index == candidate_index:
                        continue
                    similarity = population_similarity(self.islands[target_index].population, self.islands[candidate_index].population)
                    if similarity < min_similarity:
                        min_similarity = similarity
                        source_index = candidate_index

                while(len(immigrants) < removed_count):
                    index = self.islands[source_index].weighted_random_select(1)[0]
                    immigrant_mlc = self.islands[source_index].population[index]
                    if immigrant_mlc.smi not in target_smis:
                        immigrants.append(immigrant_mlc)
                    
                print(f"Island {target_index}: Removed {removed_count} individuals, Immigrated {len(immigrants)} individuals.")
                target_island.population.extend(immigrants)

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

            print("Before Immigration:")
            for island in self.islands:
                island.log_intermediate()

            self.cluster_filling_immigration(max_cluster_size=3, cut_off_threshold=0.4)

            print("After Immigration:")
            for island in self.islands:
                island.log_intermediate()
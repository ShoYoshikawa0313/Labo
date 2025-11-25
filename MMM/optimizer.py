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

    def forced_diversity_immigration(self):
        """
        多様性を強制的に向上させるための移住戦略。
        各島に対して、他の全ての島からの移住候補を評価し、
        移住後の多様性が最も高くなるような移住者の組み合わせを選択して、
        評価の低い個体と入れ替える。
        """
        if len(self.islands) <= 1:
            return
        # 各島を移住先（target）としてループ
        for target_index in range(len(self.islands)):
            # 移住先の島に存在する分子のSMILESリスト（重複チェック用）
            target_smis = [mlc.smi for mlc in self.islands[target_index].population]
            # 他の島から集めた移住候補のリスト
            candidates = []
            # 各島を移住元（source）としてループ
            for source_index in range(len(self.islands)):
                # 移住元と移住先が同じ場合はスキップ
                if target_index == source_index:
                    continue
                # 移住元の島から、移住先の島に存在しない分子をフィルタリング
                filted = []
                for mlc in self.islands[source_index].population:
                    if mlc.smi not in target_smis:
                        filted.append(mlc)

                # フィルタリング後の候補数が移住させる数に満たない場合はスキップ
                if self.immigrants_size > len(filted): continue
                # 候補を評価値でソートし、上位の個体を移住候補として選択
                filted = sorted(filted, reverse=True)[:self.immigrants_size]
                candidates.append(filted)
            
            max_diversity = -1.0
            best_candidate = None
            # 最も多様性を向上させる移住候補の組み合わせを探す
            for candidate in candidates:
                # 移住後の個体群を一時的に作成して多様性を評価
                tmp = self.islands[target_index].population[:]
                for i in range(len(candidate)):
                    tmp[(-1*i)-1] = candidate[i]
                diversity = self.islands[target_index].evaluator.diversity([mlc.smi for mlc in tmp])
                # これまでの最大多様性を超えていれば、その候補を最適候補として更新
                if diversity > max_diversity:
                    max_diversity = diversity
                    best_candidate = candidate[:]
            # 最適な移住候補が見つからなかった場合はスキップ
            if best_candidate is None:
                continue
            # 移住先の島で評価が最も低い個体から順に、最適な移住候補と入れ替える
            for i in range(len(best_candidate)):
                self.islands[target_index].population[(-1*i)-1] = best_candidate[i]
            

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

    def forced_random_immigration(self):
        source_populations = [sorted(island.population[:],reverse=True) for island in self.islands]
        
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
            
            num_imm = 0
            # 移住先のSMILESをセットに変換して重複チェックを高速化
            target_smis = {mlc.smi for mlc in self.islands[target].population}
            # 移住元の個体群をコピーして、処理中に変更できるようにする
            source_candidates = source_populations[source][:]

            while(num_imm < self.immigrants_size):
                found_immigrant = False
                for i, mlc in enumerate(source_candidates):
                    if mlc.smi not in target_smis: # 高速なセットでのチェック
                        # 評価の低い個体を上書き
                        self.islands[target].population[-1 - num_imm] = mlc
                        # 移住させた分子を移住先のSMILESセットに追加
                        target_smis.add(mlc.smi)
                        num_imm += 1
                        found_immigrant = True
                        # 処理済みの候補をリストから削除
                        del source_candidates[i]
                        break # 次の移民を探す
                # 新しい移民が見つからなかった場合はループを抜ける
                if not found_immigrant:
                    break

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
    
    def niche_filling_immigration(self):
        """
        化学的ニッチを充填することによる多様性向上戦略。
        各島に存在しない、あるいは希少な化学構造（Murcko骨格）を持つ個体を
        他の島から探し出し、評価の高い個体を移住させる。
        """
        if len(self.islands) <= 1:
            return

        # 1. 全ての島の個体からMurcko骨格を抽出し、グローバルな骨格カタログを作成
        global_scaffold_catalog = {}
        for island in self.islands:
            for mlc in island.population:
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mlc.mol, includeChirality=False)
                    if scaffold:
                        if scaffold not in global_scaffold_catalog:
                            global_scaffold_catalog[scaffold] = []
                        global_scaffold_catalog[scaffold].append(mlc)
                except:
                    continue # 骨格が生成できない分子はスキップ

        # 各島を移住先としてループ
        for target_index, target_island in enumerate(self.islands):
            # 2. 移住先の島に存在する骨格のセットを作成
            target_scaffolds = set()
            for mlc in target_island.population:
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mlc.mol, includeChirality=False)
                    if scaffold:
                        target_scaffolds.add(scaffold)
                except:
                    continue

            # 3. 移住先の島に存在しない「ニッチ」な骨格を持つ移住候補を探す
            immigrant_candidates = []
            for scaffold, mlcs in global_scaffold_catalog.items():
                if scaffold not in target_scaffolds:
                    # 他の島に由来する個体のみを候補とする
                    for mlc in mlcs:
                        # この個体がどの島に由来するかを特定するのは困難なため、
                        # ここでは単純にグローバルカタログから候補を選出する。
                        # (厳密には、その個体がtarget_island由来でないことを確認すべき)
                        immigrant_candidates.append(mlc)

            if not immigrant_candidates:
                continue

            # 4. 移住候補をスコアでソートし、上位を移民として選定
            immigrant_candidates.sort(reverse=True) # Mol_Dataはスコアで比較される
            immigrants = immigrant_candidates[:self.immigrants_size]

            # 5. 移住先の評価が最も低い個体と入れ替え
            if immigrants:
                for i in range(len(immigrants)):
                    self.islands[target_index].population[(-1*i)-1] = immigrants[i]

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

            self.forced_random_immigration()
import random
import requests
from tqdm import tqdm
from rdkit import Chem
import numpy as np

# 自作モジュールのインポート
import LLM_operator.crossover as co
from evaluator import smiles_similarity

# スコアが0になるのを防ぐための微小な値
MINIMUM = 1e-10

class BioT5:
    def __init__(self, composit):
        
        self.molleo = composit["molleo"]
        self.offspring_size = composit["offspring_size"]
        self.bin_size = composit["bin_size"]
        self.prompt_template = composit["prompt_template"]
        self.model_name = composit["LLM"]["name"]

        self.base_url = composit["LLM"]["URL"]
        self.endpoint = "/biot5/"
    
    def sanitize_smiles(self, smi):
        """
        Return a canonical smile representation of smi 
        """
        if smi is None or smi == "":
            return None
        if "." in smi:
            return None
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi_canon
        except:
            #print(f"Invalid SMILES : {smi}")
            return None

    def request(self,smi):

        params = {
        "smiles": smi,
        "task": self.prompt_template
        }

        try:
            # GETリクエストを送信
            response = requests.get(f"{self.base_url}{self.endpoint}", params=params)
            # ステータスコードをチェックして、リクエストが失敗した場合に例外を発生させる
            response.raise_for_status()
            # サーバーからの応答をJSONからPythonの辞書に変換して返す
            response_dict = response.json()
            return "BioT5", self.sanitize_smiles(response_dict["smiles"])

        except Exception as e:
            #print(f"{type(e).__name__} {e}")
            #print("Invalid Response")
            return None,None
    
    def reproduce(self, mating_list: list):
        while(True):
            parent = []
            # メイティングプールからランダムに2つの親を選択
            parent.append(random.choice(mating_list))
            parent.append(random.choice(mating_list))

            # 2つの親分子を交叉させ、新しい子分子を生成
            new_child = co.crossover(parent[0].mol, parent[1].mol)
            try:
                new_child_smi = Chem.MolToSmiles(new_child)
                if new_child_smi is not None :
                    return new_child_smi, parent[0].smi, parent[1].smi
            except:
                print("Error : Invalid crossover in reproduce")

    def weighted_random_select(self, population, size, reverse=False):
        """
        reverse=Trueの場合、スコアが低い個体を優先的に選択します。
        reverse=Falseの場合、スコアが高い個体を優先的に選択します。
        defaultはreverse=Falseです。
        """
        # スコアを抽出
        if reverse == False:
            population_scores = [mlc.score for mlc in population]
        else:
            population_scores = [1.0 - mlc.score for mlc in population]
        # スコアと分子をタプルのリストにまとめる
        all_tuples = list(zip(population_scores, population))
        # スコアに微小な値を加えて、ゼロ除算を回避する
        population_scores = [s + MINIMUM for s in population_scores]
        # スコアの合計を計算
        sum_scores = sum(population_scores)
        # 各個体のスコアを正規化し、選択確率を計算
        population_probs = [p / sum_scores for p in population_scores]
        # 計算された確率分布に基づき、個体のインデックスを復元抽出で選択
        indices = np.random.choice(len(all_tuples), p=population_probs, size=size, replace=True)
        return indices
    
    def mating(self, population, process_id=0):

        # スコアに基づいて親集団（メイティングプール）を形成
        indices = self.weighted_random_select(population, self.offspring_size)
        mating_list = [population[index] for index in indices]

        top_smi = population[0].smi

        families = []

        if not self.molleo:
            for i in tqdm(range(self.bin_size), position=process_id, desc=f"{self.model_name:<15}"):
                while(True):
                    inter_smi, parent1_smi, parent2_smi = self.reproduce(mating_list)
                    response_model, offspring_smi = self.request(inter_smi)
                    if offspring_smi is None: continue
                    families.append({"offspring":offspring_smi, "parent1":parent1_smi, "parent2":parent2_smi, "inter":inter_smi})
                    break

        else:
            for i in range(self.offspring_size):
                inter_smi, parent1_smi, parent2_smi = self.reproduce(mating_list)
                families.append({"offspring":inter_smi, "parent1":parent1_smi, "parent2":parent2_smi})
            j = 0
            while(len(families) < self.bin_size):
                parent_smi = population[j].smi
                response_model, offspring_smi = self.request(parent_smi)
                if offspring_smi is not None:
                    families.append({"offspring":offspring_smi, "parent1":parent_smi, "parent2":parent_smi})
                j += 1

        families.sort(key=lambda x: smiles_similarity(x["offspring"], top_smi), reverse=True)
        return families[:self.offspring_size]
    
        

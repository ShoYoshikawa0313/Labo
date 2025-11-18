import random
import requests
import re
from rdkit import Chem
from tqdm import tqdm

# 自作モジュールのインポート
import LLM_operator.crossover as co

class Drug_Assist:
    def __init__(self, composit):
        
        self.offspring_size = composit["offspring_size"]
        self.prompt_template = composit["prompt_template"]
        self.model_name = composit["LLM"]["name"]

        self.max_length = 100
        self.base_url = composit["LLM"]["URL"]
        self.endpoint = "/api/generate"

        self.num_try = 0
        self.num_error = 0
    
    def sanitize_smiles(self, smi):
        """
        Return a canonical smile representation of smi 
        """
        if smi is None or smi == "":
            return None
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi_canon
        except:
            #print(f"Invalid SMILES : {smi}")
            return None
        
    def response2smi(self, response):
        smi = re.findall(r'"([^"]*)"',response)[0]
        clean_smi = self.sanitize_smiles(smi)
        return clean_smi
    
    def request(self,smi):

        params = {
        "model": self.model_name,
        "prompt": self.prompt_template.replace("<<<SMILES>>>",smi),
        "keep_alive": 10,
        "stream": False,
        "options": {
            "num_predict": self.max_length
        }
        }
        try:
            # POSTリクエストを送信
            response = requests.post(f"{self.base_url}{self.endpoint}", json=params)
            # ステータスコードをチェックして、リクエストが失敗した場合に例外を発生させる
            response.raise_for_status()
            # サーバーからの応答をJSONからPythonの辞書に変換する
            response_dict = response.json()
            return response_dict["model"], self.response2smi(response_dict["response"])

        except Exception as e:
            #print(f"{type(e).__name__} {e}",flush=True)
            #print("Invalid Response",flush=True)
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
    
    def mating(self, mating_list: list):
        families = []
        log = ""
        for i in tqdm(range(self.offspring_size), desc=f"{self.model_name}  "):
            while(True):
                self.num_try += 1
                inter_smi, parent1_smi, parent2_smi = self.reproduce(mating_list)
                response_model, offspring_smi = self.request(inter_smi)
                if offspring_smi is None:
                    self.num_error += 1
                    continue
                log +=  f"\n{i} / {self.offspring_size} : {response_model} {offspring_smi}"
                families.append({"offspring":offspring_smi, "parent1":parent1_smi, "parent2":parent2_smi, "inter":inter_smi})
                break
        log += f"\n\nerror/try : {self.num_error}/{self.num_try}"
        #print(log)
        return families



import random
import requests
import re
import json
from rdkit import Chem

# 自作モジュールのインポート
from mol_data import Mol_Data

class Ollama:
    def __init__(self, composit):
        
        self.task = composit["task"]
        self.offspring_size = composit["offspring_size"]
        self.task_definition = composit["prompt_template"]
        self.model = composit["LLM"]["name"]

        self.max_length = 200
        self.base_url = "http://10.34.35.193:11434"
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
            print(f"Invalid SMILES : {smi}")
            return None
        
    def response2smi(self, response):
        contents = response.split('"SMILES"',1)[1]
        smis = re.findall(r'"([^"]*)"',contents)
        if len(smis) != 1:
            print("Invalid Contents")
            return None
        smi = smis[0]
        clean_smi = self.sanitize_smiles(smi)
        return clean_smi

    def request(self,parents_info,task,stop=False):

        params = {
        "model": self.model,
        "prompt": task.replace("<<<ParentInfo>>>",parents_info),
        "stream": False,
        "options": {
            "num_predict": self.max_length,
            "stop": ['"Explanation"']
        }
        }

        if stop: params["keep_alive"] = 0
        if self.model == "deepseek-r1:8b" or self.model == "qwen3:8b":
            params["think"] = False
        if self.model == "gpt-oss:20b":
            params["think"] = "low"

        try:
            # POSTリクエストを送信
            response = requests.post(f"{self.base_url}{self.endpoint}", json=params)
            # ステータスコードをチェックして、リクエストが失敗した場合に例外を発生させる
            response.raise_for_status()
            # サーバーからの応答をJSONからPythonの辞書に変換する
            response_dict = response.json()
            proposed_smiles = self.response2smi(response_dict["response"])
            model = response_dict["model"]
            return model,proposed_smiles

        except Exception as e:
            print(f"{type(e).__name__} {e}")
            print("Invalid Response")
            return None,None
    
    def ramdom_parents(self, mating_list):
        parent_info = ""
        parentA = random.choice(mating_list)
        parentB = random.choice(mating_list)

        parent_info += f"[ ParentA : {parentA.smi} , {parentA.score:.3f} ]\n"
        parent_info += f"[ ParentB : {parentB.smi} , {parentB.score:.3f} ]\n"
        return parent_info, parentA.smi, parentB.smi

    def mating(self, mating_list: list):
        families = []
        for i in range(self.offspring_size):
            while(True):
                self.num_try += 1
                parents_info, parent1_smi, parent2_smi = self.ramdom_parents(mating_list)
                model,offspring_smi = self.request(parents_info,self.task_definition)
                if offspring_smi is None:
                    self.num_error += 1
                    continue
                print(f"{i} / {self.offspring_size} : {model} {offspring_smi}")
                families.append(Mol_Data(self.task,Chem.MolFromSmiles(offspring_smi),offspring_smi,parent1_smi,parent2_smi))
                break
        print(f"error/try : {self.num_error}/{self.num_try}")
        return families


        





import random
import requests
import re
from rdkit import Chem
from tqdm import tqdm


class Ollama:
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
        if "." in smi:
            return None
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi_canon
        except:
            #print(f"Invalid SMILES : {smi}")
            return None
        
    def response2smi(self, response):
        contents = response.split('"SMILES"',1)[1]
        smis = re.findall(r'"([^"]*)"',contents)
        if len(smis) != 1:
            #print("Invalid Contents")
            return None
        smi = smis[0]
        clean_smi = self.sanitize_smiles(smi)
        return clean_smi

    def request(self,parents_info):

        params = {
        "model": self.model_name,
        "prompt": self.prompt_template.replace("<<<ParentInfo>>>",parents_info),
        "keep_alive": 10,
        "stream": False,
        "options": {
            "num_predict": self.max_length,
            "stop": ['"Explanation"'],   
        }
        }

        if self.model_name == "deepseek-r1:8b" or self.model_name == "qwen3:8b": params["think"] = False
        if self.model_name == "gpt-oss:20b": 
            params["think"] = "low"
            params["options"]["num_predict"] = 256

        try:
            # POSTリクエストを送信
            response = requests.post(f"{self.base_url}{self.endpoint}", json=params)
            # ステータスコードをチェックして、リクエストが失敗した場合に例外を発生させる
            response.raise_for_status()
            # サーバーからの応答をJSONからPythonの辞書に変換する
            response_dict = response.json()
            return response_dict["model"], self.response2smi(response_dict["response"])

        except Exception as e:
            #print(f"{type(e).__name__} {e}")
            #print("Invalid Response")
            return None,None
    
    def ramdom_parents(self, mating_list):
        parent_info = "\n"
        parentA = random.choice(mating_list)
        parentB = random.choice(mating_list)

        parent_info += f"[ ParentA : {parentA.smi} , {parentA.score:.3f} ]\n"
        parent_info += f"[ ParentB : {parentB.smi} , {parentB.score:.3f} ]\n"
        return parent_info, parentA.smi, parentB.smi

    def mating(self, mating_list, process_id=0):
        families = []
        log = ""
        for i in tqdm(range(self.offspring_size), position=process_id, desc=f"{self.model_name:<15}"):
            while(True):
                self.num_try += 1
                parents_info, parent1_smi, parent2_smi = self.ramdom_parents(mating_list)
                response_model,offspring_smi = self.request(parents_info)
                if offspring_smi is None:
                    self.num_error += 1
                    continue
                log +=  f"\n{i} / {self.offspring_size} : {response_model} {offspring_smi}"
                families.append({"offspring":offspring_smi, "parent1":parent1_smi, "parent2":parent2_smi})
                break
        log += f"\n\nerror/try : {self.num_error}/{self.num_try}"
        #print(log)
        return families

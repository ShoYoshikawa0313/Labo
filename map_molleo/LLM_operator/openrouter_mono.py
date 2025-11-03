from openai import OpenAI
import json
import random
import time
import re
from rdkit import Chem
from mol_data import Mol_Data

MINIMUM = 1e-10

class Open_Router: 
    def __init__(self, composit):

        self.model = composit["LLM"]
        self.request_interval = composit["interval"]

        self.task = composit["task"]
        self.prompt_template = composit["prompt_template"]
        self.offspring_size = composit["offspring_size"]

        self.client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
        )

        self.last_request_time = time.time()
        
    def request(self, question):
        message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]
        message.append({"role": "user", "content": question})

        response = None
        while(True):
            now = time.time()
            if now - self.last_request_time > self.request_interval:
                try:
                    response = self.client.chat.completions.create(
                    model= self.model,
                    messages = message
                    ) 
                except Exception as e:
                    print(f"{type(e).__name__} {e}")
                    print(f"request error")
                    return None
                self.last_request_time = now
                return response.choices[0].message.content               
            else:
                time.sleep(0.5)

    def sanitize_smiles(self, smi):
        """
        Return a canonical smile representation of smi 
        """
        if smi == '':
            print("No SMILES in JSON")
            return None
        smi = smi.replace("\\\\","\\")
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi_canon
        except:
            return None

    def response2smis(self, response):

        # レスポンス文字列から開始マーカー "<<<json start>>>" を削除します。
        response = response.replace("<<<json start>>>", "")
        # 同様に、終了マーカー "<<<json end>>>" を削除します。
        response = response.replace("<<<json end>>>", "")
        # 文字列の先頭および末尾にある不要な空白（スペースや改行など）を除去します。
        response = response.strip()
        # JSONとして正しく解析できるように、文字列中のバックスラッシュ `\` を `\\` にエスケープ（置換）します。
        response_json = response.replace("\\","\\\\")

        data = None
        try:
            data = json.loads(response_json)
        except Exception as e:
            print(f"{type(e).__name__} {e}")
            print("Invalid JSON Error")
            return None

        smi = data["SMILES"] if data["SMILES"] is not None else ""
        smi = self.sanitize_smiles(smi)
        return smi

    def ramdom_parents(self, mating_list):
        parents = []
        parent_infos = []
        for i in range(self.offspring_size):
            parentA = random.choice(mating_list)
            parentB = random.choice(mating_list)
            parents.append((parentA,parentB))

            parent_info = ""
            parent_info += f"[ ParentA : {parentA.smi} , {parentA.score:.3f} ]\n"
            parent_info += f"[ ParentB : {parentB.smi} , {parentB.score:.3f} ]\n"

            parent_infos.append(parent_info)
        return parent_infos, parents
    
    def mating(self, mating_list: list):
        families = []
        parent_infos, parents = self.ramdom_parents(mating_list)
        for i in range(self.offspring_size):
            prompt = self.prompt_template.replace("<<<ParentInfo>>>",parent_infos[i])

            response = self.request(prompt)
            if response is None : 
                print("Invalid Response")
                continue

            smi = self.response2smis(response)
            if smi is None : 
                print("Invalid SMILES")
                continue

            print(f"{i} / {self.offspring_size} : {smi}")
            families.append(Mol_Data(self.task,Chem.MolFromSmiles(smi),smi,parents[i][0].smi,parents[i][1].smi))
            
        return families


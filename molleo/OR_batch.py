from openai import OpenAI
import re
import yaml
import json
import random
import time

from rdkit import Chem

MINIMUM = 1e-10

class OPEN_ROUTER: 
    def __init__(self, args, model, interval):

        self.args = args

        self.model = model
        self.request_interval = interval

        self.client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
        )

        self.prompt = None
        with open(self.args.LLM_prompt, 'r', encoding='utf-8') as f:
            self.prompt = yaml.safe_load(f)["LLM"]["original_batch"]

        self.head = self.prompt["head"]
        self.tail = self.prompt["tail"]

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
            print("Invalid JSON Error")
            return None

        smis = []
        for pair in data.keys():
            smi = data[pair]["SMILES"]
            smi = self.sanitize_smiles(smi)
            if smi is not None: smis.append(smi)
            else: smis.append("")

        if len(smis) != self.args.offspring_size:
            print("Not enough SMILES Error")
            return None

        return smis

    def ramdom_parents(self, mating_list):
        parents = []
        parent_info = ""
        for i in range(self.args.offspring_size):
            parentA = random.choice(mating_list)
            parentB = random.choice(mating_list)
            parents.append((parentA,parentB))

            parent_info += f"Pair {i+1}:\n"
            parent_info += f"[ ParentA : {parentA.smi} , {parentA.score:.3f} ]\n"
            parent_info += f"[ ParentB : {parentB.smi} , {parentB.score:.3f} ]\n"
        return parent_info, parents

    def generational_shift(self, mating_list: list):
        while(True):
            parent_info, parents = self.ramdom_parents(mating_list)
            prompt = self.head + parent_info + self.tail

            response = self.request(prompt)
            if response is None : continue

            smis = self.response2smis(response)
            if smis is None : continue

            families = []
            for i, smi in enumerate(smis):
                if smi != "":
                    print(f"{i} / {len(smis)} {smi}")
                    families.append((smi, parents[i][0].smi, parents[i][1].smi))
                else: print(f"{i} / {len(smis)} Invalid Smiles")

            return families


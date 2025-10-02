from openai import OpenAI
import re
import yaml
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
            self.prompt = yaml.safe_load(f)["LLM"]["original"]

        self.requirements = self.prompt["requirements"]
        self.task_definition = self.prompt["task2description"]
        self.task_objective = self.prompt["task2objective"]

        self.last_request_time = time.time()
        
    def sanitize_smiles(self, smi):
        """
        Return a canonical smile representation of smi 
        """
        if smi == '':
            return None
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi_canon
        except:
            return None
        
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

    def generational_shift(self, mating_list: list):
        families = []
        for i in range(self.args.offspring_size):
            
            parent = []
            parent.append(random.choice(mating_list))
            parent.append(random.choice(mating_list))

            parent_info = ''
            for j in range(2):
                parent_info += '\n[' + parent[j].smi + ',' + str(parent[j].score) + ']'

            prompt = self.task_definition + parent_info + self.task_objective + self.requirements

            try:
                response = self.request(prompt)
                proposed_smiles = re.search(r'\\box\{(.*?)\}', response).group(1)
                smi = self.sanitize_smiles(proposed_smiles)
                families.append((smi,parent[0].smi,parent[1].smi))
            except Exception as e:
                print(f"{type(e).__name__} {e}")
                print("Invalid Response")
                continue

        return families


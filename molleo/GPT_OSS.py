from openai import OpenAI
import re
import yaml
import random

from rdkit import Chem

MINIMUM = 1e-10

class GPT_OSS:
    def __init__(self, args):

        self.args = args

        self.gpt = OpenAI(
            base_url = 'http://10.34.35.194:11434/v1',
            api_key='ollama', # required, but unused
        )

        self.prompt = None
        with open(self.args.LLM_prompt, 'r', encoding='utf-8') as f:
            self.prompt = yaml.safe_load(f)["GPT"]

        self.requirements = self.prompt["requirements"]
        self.task_definition = self.prompt["task2description"][self.args.task]
        self.task_objective = self.prompt["task2objective"][self.args.task]
        
        
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
        
    def gpt_request(self, question, temperature=0.0):
        message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]

        message.append({"role": "user", "content": question})

        params = {
            "model": "gpt-oss:20b",
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": message
        }

        for retry in range(3):
            try:
                response = self.gpt.chat.completions.create(**params).choices[0].message.content
                message.append({"role": "assistant", "content": response})
                break
            except Exception as e:
                print(f"{type(e).__name__} {e}")

        return message, response

    def edit_smi(self, parent_info: str):
        
        while(True):
            try:
                prompt = self.task_definition + parent_info + self.task_objective + self.requirements

                message, response = self.gpt_request(self.gpt,prompt)
                proposed_smiles = re.search(r'\\box\{(.*?)\}', response).group(1)
                proposed_smiles = self.sanitize_smiles(proposed_smiles)
                
                if proposed_smiles is not None :  return proposed_smiles

            except Exception as e:
                print( '\033[31m' + "Error Invalid Response!! Retry !!" + '\033[0m' )

    def generational_shift(self, mating_list: list):
        families = []
        for i in range(self.args.offspring_size):
            
            parent = []
            parent.append(random.choice(mating_list))
            parent.append(random.choice(mating_list))

            parent_info = ''
            for j in range(2):
                parent_info += '\n[' + parent[j].smi + ',' + str(parent[j].score) + ']'

            edited_smi = self.edit_smi(parent_info)
            families.append((edited_smi,parent[0].smi,parent[1].smi))

        return families


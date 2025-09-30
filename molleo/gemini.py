import google.generativeai as genai
import re
import yaml
import json
import random
import time

from rdkit import Chem

MINIMUM = 1e-10

genai.configure(api_key="AIzaSyB1yPa0EsQ21nfyVy_1uxm1UACtgkh_1aE")

class Gemini:
    def __init__(self, args):

        self.args = args

        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")

        self.prompt = None
        with open(self.args.LLM_prompt, 'r', encoding='utf-8') as f:
            self.prompt = yaml.safe_load(f)["GPT_Batch"]

        self.requirements = self.prompt["requirements"]
        self.task2objective = self.prompt["task2objective"]
        
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
        
    def gpt_request(self, prompt):

        for retry in range(3):
            try:
                response = self.model.generate_content(prompt).text
                break
            except Exception as e:
                print(f"{type(e).__name__} {e}")
                print(f"gemini failed")

        return prompt, response

    def edit_smi(self, parent_info: str):
        
        response_json = ""
        data = None
        while(True):
            try:
                prompt = self.requirements + parent_info + self.task2objective
                message, response = self.gpt_request(prompt)
                response_json = response.replace("<<<json start>>>", "").replace("<<<json end>>>", "").strip()
                data = json.loads(response_json)
                break

            except Exception as e:
                print(f"{type(e).__name__} {e}")
                print( '\033[31m' + "Error Invalid Response!! Retry !!" + '\033[0m' )
        
        offspring_smis = []
        for pair in data.keys():
            smi = data[pair]["SMILES"]
            smi = self.sanitize_smiles(smi)
            if smi is not None: offspring_smis.append(smi)
            else: offspring_smis.append("")
        
        return offspring_smis

    def generational_shift(self, mating_list: list):
        parents = []
        parent_info = ""
        for i in range(self.args.offspring_size):

            parentA = random.choice(mating_list)
            parentB = random.choice(mating_list)
            parents.append((parentA,parentB))

            parent_info += f"Pair {i+1}:\n"
            parent_info += f"[ ParentA : {parentA.smi} , {parentA.score:.3f} ]\n"
            parent_info += f"[ ParentB : {parentB.smi} , {parentB.score:.3f} ]\n"

        edited_smis = self.edit_smi(parent_info)
        families = []
        for i, edited_smi in enumerate(edited_smis):
            if edited_smi != "":
                print(f"{i} / {len(edited_smis)} {edited_smi}")
                families.append((edited_smi, parents[i][0], parents[i][1]))
            else: print(f"{i} / {len(edited_smis)} Invalid Smiles")

        return families


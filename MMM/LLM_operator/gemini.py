import google.generativeai as genai
import random
import re
from rdkit import Chem
import time

MINIMUM = 1e-10

genai.configure(api_key="AIzaSyA65cJWE4vjnoXs4m5LRRBw_QW3-mk2Vko")

class Gemini:
    def __init__(self, composit):
        self.model = genai.GenerativeModel(composit["LLM"]["name"])
        self.request_interval = composit["interval"]

        self.offspring_size = composit["offspring_size"]
        self.prompt_template = composit["prompt_template"]
        self.model_name = composit["LLM"]["name"]

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
        proposed_smis = []
        if response is None : return None
        if response.count('"SMILES') != self.offspring_size or response.count('"Explanation"') != self.offspring_size: return None
        contents = response.split('"SMILES"')[1:]
        for content in contents:
            middle = content.split('"Explanation"')[0]
            print(middle)
            smi = re.findall(r'"([^"]*)"',middle)[0]
            clean_smi = self.sanitize_smiles(smi)
            if clean_smi is None: proposed_smis.append("")
            else : proposed_smis.append(clean_smi)
        return proposed_smis


    def request(self, parent_info):
        time.sleep(self.request_interval)
        try:
            print("request")
            response = self.model.generate_content(self.prompt_template.replace("<<<ParentInfo>>>",parent_info)).text
            return "gemini",self.response2smi(response)
        except Exception as e:
            print(f"{type(e).__name__} {e}")
            print("Invalid Response")
            return None,None

    def ramdom_parents(self, mating_list):
        parents = []
        parent_info = ""
        for i in range(self.offspring_size):
            parentA = random.choice(mating_list)
            parentB = random.choice(mating_list)
            parents.append((parentA,parentB))

            parent_info += f"Pair {i+1}:\n"
            parent_info += f"[ ParentA : {parentA.smi} , {parentA.score:.3f} ]\n"
            parent_info += f"[ ParentB : {parentB.smi} , {parentB.score:.3f} ]\n"
        return parent_info, parents

    def mating(self, mating_list: list):
        while(True):
            parent_info, parents = self.ramdom_parents(mating_list)
            response_model, offspring_smis = self.request(parent_info)
            if offspring_smis is None :
                input()
                continue

            families = []
            log = ""
            for i, offspring_smi in enumerate(offspring_smis):
                self.num_try+=1
                if offspring_smi != "":
                    log +=  f"\n{i} / {self.offspring_size} : {response_model} {offspring_smi}"
                    families.append({"offspring":offspring_smi, "parent1":parents[i][0].smi, "parent2":parents[i][1].smi})
                else:
                    self.num_error += 1
                    log +=  f"\n{i} / {self.offspring_size} : Invalid SMILES"
            log += f"\n\nerror/try : {self.num_error}/{self.num_try}"
            print(log)
            return families


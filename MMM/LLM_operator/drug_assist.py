import random
from openai import OpenAI
import re
from rdkit import Chem

# 自作モジュールのインポート
import LLM_operator.crossover as co
from mol_data import Mol_Data

from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

class Drug_Assist:
    def __init__(self, composit):
        
        self.task = composit["task"]
        self.offspring_size = composit["offspring_size"]
        self.task_definition = composit["prompt_template"]

        self.client = OpenAI(
            base_url = 'http://10.34.35.193:11434/v1',
            api_key='ollama', # required, but unused
        )
    
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

    def drug_assist_request(self,smi,task):

        prompt = task.replace("<<<SMILES>>>",smi)

        params = {
            "model": "drugassist-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
        }

        try:
            # リクエストを送信
            response = self.client.chat.completions.create(**params).choices[0].message.content
            return re.findall(r'"([^"]*)"',response)[0]

        except Exception as e:
            print("Invalid Response")

        return ""

    def edit_smi(self, smi):
        response = self.drug_assist_request(smi, self.task_definition)
        proposed_smiles = self.sanitize_smiles(response)

        if proposed_smiles is not None: return proposed_smiles
        else:
            print("Invalid mutation.")
            return smi
    
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
    
    def similarity(self, smi1, smi2):
        fp_smi1 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi1), 2, nBits=1024)
        fp_smi2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi2), 2, nBits=1024)
        return TanimotoSimilarity(fp_smi1,fp_smi2)
    
    def mating(self, mating_list: list):
        families = []
        for i in range(self.offspring_size):
            while(True):
                inter_smi, parent1_smi, parent2_smi = self.reproduce(mating_list)
                offspring_smi = self.edit_smi(inter_smi)
                if offspring_smi is None: continue
                print(f"{i} / {self.offspring_size} : {inter_smi} => {offspring_smi} {self.similarity(inter_smi,offspring_smi)}")
                families.append(Mol_Data(self.task,Chem.MolFromSmiles(offspring_smi),offspring_smi,parent1_smi,parent2_smi,inter_smi))
                break
        return families
    
    def test(self,parents):
        new_child = co.crossover(parents[0].mol, parents[1].mol)
        new_child_smi = Chem.MolToSmiles(new_child) if Chem.MolToSmiles(new_child) is not None else parents[0].smi

        response = self.drug_assist_request(new_child_smi, self.task_definition)
        if response == "":return "RESPONSE"
        proposed_smiles = self.sanitize_smiles(response)
        if proposed_smiles is None:return "SMILES"

        return proposed_smiles,new_child_smi,self.similarity(proposed_smiles,new_child_smi)



import random
import requests
import yaml

from rdkit import Chem

# 自作モジュールのインポート
import crossover as co

class BioT5:
    def __init__(self, args):
        
        self.args = args

        self.base_url = "http://10.34.35.194:5000"
        self.endpoint = "/biot5/"

        self.prompt = None
        with open(self.args.LLM_prompt, 'r', encoding='utf-8') as f:
            self.prompt = yaml.safe_load(f)["BioT5"]

        self.task_definition = self.prompt["task2description"][self.args.task]
    
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

    def biot5_request(self,smi,task):

        params = {
        "smiles": smi,
        "task": task
        }

        try:
            # GETリクエストを送信
            response = requests.get(f"{self.base_url}{self.endpoint}", params=params)
            # ステータスコードをチェックして、リクエストが失敗した場合に例外を発生させる
            response.raise_for_status()
            # サーバーからの応答をJSONからPythonの辞書に変換して返す
            return response.json()["smiles"]

        except requests.exceptions.HTTPError as http_err:
            print(f"An HTTP error occurred: {http_err}")
            print(f"Response body: {response.text}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"A connection error occurred: {conn_err}")
            print("Please check if the Flask server is running and the URL is correct.")
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected request error occurred: {req_err}")

        return ""

    def edit_smi(self, smi):
        response = self.biot5_request(smi, self.task_definition)
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
    
    def generational_shift(self, mating_list: list):
        families = []
        for i in range(self.args.offspring_size):
            # Step 1 交叉という標準的な遺伝的操作を用いて、ベースとなる子孫集団を生成
            # メイティングプールから親を選択し、交叉を繰り返して指定された数の子孫候補を生成します。
            base_smi, parent1_smi, parent2_smi = self.reproduce(mating_list)
            # Step 2 スコアが上位の優れた親分子をBioT5モデルに入力し、より有望な化学構造空間を探索するために分子を「編集」させる
            # BioT5モデルで編集させます。
            edited_smi = self.edit_smi(base_smi)
            print(f"{i} / {self.args.offspring_size} : {parent1_smi}, {parent2_smi} => {base_smi}")
            families.append((edited_smi, parent1_smi, parent2_smi))
            
        return families


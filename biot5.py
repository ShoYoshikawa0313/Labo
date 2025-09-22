import random
import requests
import json
import numpy as np

import selfies as sf

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

# 自作モジュールのインポート
import crossover as co

class CLM:
    def __init__(self):

        self.base_url = "http://10.34.35.194:5000"
        self.endpoint = "/biot5/"

        self.task2description = {
                'qed': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like a drug.\n\n',
                'jnk3': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that is a greater inhibitor of JNK3.\n\n',
                'drd2': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that inhibits DRD2 more.\n\n',
                'gsk3b': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that inhibits GSK3B more.\n\n',
                'Isomers_C9H10N2O2PF2Cl': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that has the formula C9H10N2O2PF2Cl.\n\n',
                'perindopril_mpo': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like Perindopril.\n\n',
                'sitagliptin_mpo': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like Sitagliptin.\n\n',
                'ranolazine_mpo': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like Ranolazine.\n\n',
                'thiothixene_rediscovery': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like Thiothixene.\n\n',
                'mestranol_similarity': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like Mestranol.\n\n',
                }

    def send_biot5_request(self,smiles,task):

        params = {
        "smiles": smiles,
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


    def edit(self, smiles, tasks):
        task_definition = self.task2description[tasks[0]]

        proposed_smiles = self.send_biot5_request(smiles, task_definition)
        proposed_smiles = sanitize_smiles(proposed_smiles)

        if proposed_smiles is not None: return proposed_smiles
        return None
    
class BioT5:

    def __init__(self, args):

        self.args = args
        self.clm = CLM()
    
    def get_best_smiles(self, population_scores, population_mol):
        '''
        現在の分子集団の中から、最も高いスコアを持つ分子を特定し、そのSMILES表現を返します。

        Args:
            population_scores (list): 集団内の各分子のスコアリスト。
            population_mol (List[Mol]): 現在の分子集団（RDKitのMolオブジェクトのリスト）。

        Returns:
            str: 最もスコアの高い分子のSMILES文字列。
        '''
        best_mol_idx = np.argmax(population_scores)
        top_mol = population_mol[best_mol_idx]
        top_smi = Chem.MolToSmiles(top_mol)
        return top_smi

    def get_fp_scores(self, smiles_back, target_smi):
        smiles_back_scores = []
        target = Chem.MolFromSmiles(target_smi)
        fp_target = AllChem.GetMorganFingerprint(target, 2)
        for item in smiles_back:
            mol = Chem.MolFromSmiles(item)
            fp_mol = AllChem.GetMorganFingerprint(mol, 2)
            score = TanimotoSimilarity(fp_mol, fp_target)
            smiles_back_scores.append(score)
        return smiles_back_scores
    
    def reproduce(self, mating_tuples):
        while(True):
            parent = []
            # メイティングプールからランダムに2つの親を選択
            parent.append(random.choice(mating_tuples))
            parent.append(random.choice(mating_tuples))

            # 親のタプルからMolオブジェクトのみを抽出
            parent_mol = [t[1] for t in parent]
            # 2つの親分子を交叉させ、新しい子分子を生成
            new_child = co.crossover(parent_mol[0], parent_mol[1])
            
            try:
                new_child_smi = Chem.MolToSmiles(new_child)
                if new_child_smi is not None :  return new_child_smi
            except:
                print("Error : Invalid crossover in reproduce")
    
    def generate(self, population_scores, population_mol, mating_tuples):

        top_smi = self.get_best_smiles(population_scores, population_mol)

        # Step 1 交叉という標準的な遺伝的操作を用いて、ベースとなる子孫集団を生成
        # メイティングプールから親を選択し、交叉を繰り返して指定された数の子孫候補を生成します。
        base_smis = [self.reproduce(mating_tuples) for _ in range(self.args.offspring_size)]

        # Step 2 スコアが上位の優れた親分子をBioT5モデルに入力し、より有望な化学構造空間を探索するために分子を「編集」させる
        
        offspring_smi = []
        for i, smi in enumerate(base_smis):
            # BioT5モデルで編集させます。
            edited = self.clm.edit(smi, self.args.tasks)
            print(f"{i} / {self.args.offspring_size} : {edited}")
            # 編集が成功し、有効な分子が生成された場合
            if edited is not None: offspring_smi.append(edited)
        
        # 集めた全ての分子候補の中から、「現世代で最も優れた分子（top_smi）」に構造が類似しているものを選択する

        # 全ての候補分子とトップスコア分子との間の分子指紋（fingerprint）に基づく類似度を計算します。
        sim = self.get_fp_scores(offspring_smi, top_smi)
        
        # 類似度スコアを降順にソートし、最終的な子孫集団のサイズに相当する数のインデックスを取得します。
        sorted_idx = np.argsort(np.squeeze(sim))[::-1][:self.args.offspring_size]
        
        # 取得したインデックスに基づき、類似度の高いSMILESを選択します。
        offspring_smi = np.array(offspring_smi)[sorted_idx].tolist()
        
        # 最終的に選択されたSMILESをMolオブジェクトに変換し、次世代の集団とします。
        return [Chem.MolFromSmiles(s) for s in offspring_smi]

def sanitize_smiles(smi):
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

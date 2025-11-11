import os
import random
import yaml
import math
import numpy as np  # 数値計算に使用
import time
import matplotlib.pyplot as plt
from rdkit import Chem, rdBase  # 分子操作のためのRDKitライブラリ
rdBase.DisableLog('rdApp.error')  # RDKitのエラーログを無効化

from tdc import Oracle, Evaluator
from tdc.generation import MolGen # tdc.generation.MolGenクラスをインポートします。分子生成タスクのためのデータセットをロードします。

from LLM_operator.biot5 import BioT5
from LLM_operator.drug_assist import Drug_Assist
from LLM_operator.llaSMol import LlaSMol
from LLM_operator.llama import Llama
from LLM_operator.deepseek import DeepSeek
from LLM_operator.gemma import Gemma
from LLM_operator.mistral import Mistral

from mol_data import Mol_Data

# スコアが0になるのを防ぐための微小な値
MINIMUM = 1e-10

class Evaluator():
    def __init__(self, model_name):

        composition = None
        with open("composition.yaml", 'r', encoding='utf-8') as f:
            composition = yaml.safe_load(f)[model_name][0]
        
        print(composition)

        ###変更箇所###
        self.LLM = LlaSMol(composition)

        self.task_evaluator = Oracle(name = "qed")
        self.task = "qed"

        data = MolGen(name = 'ZINC') # ZINCデータセットをロードします。
        self.all_smiles = data.get_data()['smiles'].tolist() # データセットからSMILESのリストを取得します。
        self.all_smiles.sort()

    def score_smi(self,smi):
        # SMILES文字列がNoneの場合、スコア0を返します。
        if smi is None:
            return 0
        # SMILES文字列からRDKitのMolオブジェクトを生成します。
        mol = Chem.MolFromSmiles(smi)
        # Molオブジェクトが生成できない、またはSMILES文字列が空の場合、スコア0を返します。
        if mol is None or len(smi) == 0:
            return 0
        else:
            # Molオブジェクトを正規化されたSMILES文字列に変換します。
            smi = Chem.MolToSmiles(mol)
            # 評価器を使用して分子のフィットネス（適合度）を計算します。
            fitness = float(self.task_evaluator(smi))
            # フィットネスがNaN（非数）の場合、0に設定します。
            if math.isnan(fitness):
                fitness = 0
            # オラクルの名前に "docking" が含まれている場合、フィットネスの符号を反転させます。

            # 分子バッファから該当するSMILES文字列のフィットネスを返します。
            return fitness

    def evaluate(self,num):

        output_dir = "results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        scores = []
        similarities = []
        times = []
        smiles_error = 0
        response_error = 0

        for i in range(num):

            print(f"{i} / {num}")
            
            parents_smi = np.random.choice(self.all_smiles, 2).tolist() 
            parents = []
            for smi in parents_smi:
                mol = Chem.MolFromSmiles(smi)
                parents.append(Mol_Data(self.task, mol, smi))

            start = time.time()
            response = self.LLM.test(parents)
            print(response)
            end = time.time()
            times.append(end-start)

            if len(response) != 3:
                if response == "RESPONSE": response_error+=1
                if response == "SMILES": smiles_error+=1

            elif len(response) == 3:
                child_smi,base_smi,similarity = response
                similarities.append(similarity)
                scores.append(self.score_smi(child_smi) - self.score_smi(base_smi))

        print(f"smiles error = {smiles_error}, response error = {response_error}")
        
        print(f"smiles error = {smiles_error}, response error = {response_error}")
        
        # 1. Similaritiesの分布 (データ: similarities, range: 0 to 1)
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=20, color='skyblue', edgecolor='black', range=(0, 1))
        plt.title('Distribution of Similarities')
        plt.xlabel('Similarity')
        plt.ylabel('Frequency')
        plt.xticks([i / 20 for i in range(21)]) 
        plt.grid(axis='y', alpha=0.75)
        # 変更箇所: パスを "results/similarities.png" に変更
        plt.savefig(os.path.join(output_dir, "similarities.png")) 

        # 2. Timeの分布 (データ: times, range: 0から5秒)
        plt.figure(figsize=(10, 6))
        plt.hist(times, bins=20, color='skyblue', edgecolor='black', range=(0, 5)) 
        plt.title('Distribution of Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.xticks([i * 0.25 for i in range(21)]) 
        plt.grid(axis='y', alpha=0.75)
        # 変更箇所: パスを "results/times.png" に変更
        plt.savefig(os.path.join(output_dir, "times.png"))

        # 3. Score Deltaの分布 (データ: scores, range: -1から1)
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, color='skyblue', edgecolor='black', range=(-1, 1)) 
        plt.title('Distribution of Score Delta')
        plt.xlabel('Score Delta (Child Score - Base Score)')
        plt.ylabel('Frequency')
        plt.xticks([i * 0.1 - 1.0 for i in range(21)]) 
        plt.grid(axis='y', alpha=0.75)
        # 変更箇所: パスを "results/scores.png" に変更
        plt.savefig(os.path.join(output_dir, "scores.png"))

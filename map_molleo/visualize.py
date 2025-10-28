import yaml
import glob
import os
import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statistics import mean
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from rdkit.DataStructs import TanimotoSimilarity
from tdc import Evaluator

from mol_data import Mol_Data

def results_load(input_dir, island):
    # 指定されたディレクトリ内で'population_'で始まるすべてのYAMLファイルを検索します。
    yaml_files = glob.glob(os.path.join(input_dir, 'population_*.yaml'))
    island.num_generation = len(yaml_files)

    def get_generation_number(filepath):
        """
        ファイル名から世代番号を抽出するためのヘルパー関数です。
        """
        filename = os.path.basename(filepath)
        # ファイル名の形式は 'population_{世代番号}G.yaml' を想定しています。
        try:
            # 世代番号の文字列を抽出し、整数に変換します。
            gen_str = filename.split('_')[1].replace('G.yaml', '')
            return int(gen_str)
        except (IndexError, ValueError):
            # ファイル名が期待される形式でない場合は-1を返します。
            return -1

    # 世代番号に基づいてYAMLファイルをソートし、時系列順に処理できるようにします。
    yaml_files.sort(key=get_generation_number)

    # ソートされたファイルリストをループ処理し、YAMLデータを読み込んで結果リストに追加します。
    print("Loading results ...")
    for yaml_file in tqdm.tqdm(yaml_files):
        with open(yaml_file, 'r', encoding='utf-8') as f:
            datas = yaml.safe_load(f) #datas : {smi, Mol_data} の世代集合
            population = []
            for key in datas.keys():
                data = datas[key]
                population.append(Mol_Data(Chem.MolFromSmiles(data["smi"]),data["smi"],data["score"],data["parent1_smi"],data["parent2_smi"]))
            island.populations.append(population)

            for mlc in population:
                if mlc.smi not in island.smi2mlc:
                    island.smi2mlc[mlc.smi] = mlc

class Visualizer:
    """
    遺伝的アルゴリズムによる最適化の結果を可視化するためのクラスです。
    YAMLファイルから世代ごとの集団データを読み込み、プロット機能を提供します。
    """
    def __init__(self):
        """
        Visualizerクラスを初期化します。
        """
        self.output_root_dir = "visual_results"
        if not os.path.exists(self.output_root_dir):
                os.mkdir(self.output_root_dir)

        self.populations = []
        self.populations_sub = []
        self.smi2mlc = {}
        self.num_generation = 0

        self.diversity_evaluator = Evaluator(name = 'Diversity')
            
    def plot_score_shift(self, freq):

        print("start plot score shift ...")

        output_path = os.path.join(self.output_root_dir,"score_shift.png")

        plt.figure(figsize=(12, 8))
        # 各世代のデータについて、スコアの分布をプロットします。
        for i, population in enumerate(self.populations):
            if i % freq == 0:
                # YAMLから読み込まれた辞書のバリューがスコアです。
                scores = [mlc.score for mlc in population]
                sns.kdeplot(scores, label=f'G{i}')

        plt.legend()
        plt.savefig(output_path)

    def plot_diversity_shift(self):
        
        print("start plot diversity shift ...")

        output_path = os.path.join(self.output_root_dir,"diversity_shift.png")

        diversities = []
        for population in self.populations:
            smis = [mlc.smi for mlc in population]
            diversities.append(self.diversity_evaluator(smis))

        plt.figure(figsize=(12, 8))
        plt.plot(range(len(diversities)), diversities, marker='o')
        plt.title('Diversity over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.grid(True)
        plt.savefig(output_path)

    def visualize_crossover(self):

        print("start visualize crossover ...")

        output_dir = os.path.join(self.output_root_dir,"visual_crossover")
        if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        
        num_G = 0
        for population in tqdm.tqdm(self.populations):
            output_gene_dir = os.path.join(output_dir, f"population_{num_G}G")
            if not os.path.exists(output_gene_dir):
                os.mkdir(output_gene_dir)

            for mlc in population:
                if mlc.parent1_smi != "" and mlc.parent2_smi != "":
                    offspring = mlc
                    parent1 = self.smi2mlc[offspring.parent1_smi]
                    parent2 = self.smi2mlc[offspring.parent2_smi]

                    fp_parent1 = AllChem.GetMorganFingerprintAsBitVect(parent1.mol, 2, nBits=1024)
                    fp_parent2 = AllChem.GetMorganFingerprintAsBitVect(parent2.mol, 2, nBits=1024)
                    fp_offspring = AllChem.GetMorganFingerprintAsBitVect(offspring.mol, 2, nBits=1024)

                    sim_p1_off = TanimotoSimilarity(fp_parent1, fp_offspring)
                    sim_p2_off = TanimotoSimilarity(fp_parent2, fp_offspring)

                    family = [parent1.mol, parent2.mol, offspring.mol]
                    legends = [f"parent1 score : {parent1.score:.3f}",
                               f"parent2 score : {parent2.score:.3f}",
                               f"offspring score : {offspring.score:.3f}\n" +
                               f"sim to p1: {sim_p1_off:.3f}, sim to p2: {sim_p2_off:.3f}"]

                    img = Draw.MolsToGridImage(family, molsPerRow=3, subImgSize=(300, 300), legends=legends)
                    
                    output_path = os.path.join(output_gene_dir,f"{offspring.smi}.png")
                    img.save(output_path)
            num_G += 1

    def calculate_population_similarity(self, population1, population2):
        """
        2つのpopulation間の類似度を「最大類似度の平均」を用いて計算します。
        具体的には、一方の集団の各個体について、もう一方の集団における最も類似度の高い個体との類似度（最大類似度）を求め、
        それらの平均値を取ります。これを双方向で行い、最終的に2つの平均値をさらに平均して返します。
        類似度計算にはMorganフィンガープリントとTanimoto係数を使用します。

        Args:
            population1 (list): Mol_Dataオブジェクトのリスト。
            population2 (list): Mol_Dataオブジェクトのリスト。

        Returns:
            float: 2つのpopulation間の平均最大類似度。
        """
        mols1 = [mol_data.mol for mol_data in population1 if mol_data.mol is not None]
        mols2 = [mol_data.mol for mol_data in population2 if mol_data.mol is not None]

        if not mols1 or not mols2:
            return 0.0

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols1]
        fps2 = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols2]

        def get_avg_max_similarity(fps_a, fps_b):
            total_max_similarity = 0
            if not fps_a or not fps_b:
                return 0.0
            
            for fp_a in fps_a:
                max_sim = 0
                for fp_b in fps_b:
                    sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
                    if sim > max_sim:
                        max_sim = sim
                total_max_similarity += max_sim
            return total_max_similarity / len(fps_a)

        avg_max_sim_1_to_2 = get_avg_max_similarity(fps1, fps2)
        avg_max_sim_2_to_1 = get_avg_max_similarity(fps2, fps1)

        return (avg_max_sim_1_to_2 + avg_max_sim_2_to_1) / 2.0
    
    def plot_similarity(self, target_island):

        print("start plot similarity ...")

        similarities = []

        max_generation_len = max(self.num_generation, target_island.num_generation)

        for idx in range(max_generation_len):
            population1 = self.populations[idx] if self.num_generation > idx else self.populations[-1] 
            population2 = target_island.populations[idx] if target_island.num_generation > idx else target_island.populations[-1]
            similarities.append(self.calculate_population_similarity(population1,population2)) 

        output_path = os.path.join(self.output_root_dir,"similarity.png")

        plt.figure(figsize=(12, 8))
        plt.plot(range(len(similarities)), similarities, marker='o')
        plt.title('Similarity over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Similarity')
        plt.grid(True)
        plt.savefig(output_path)


# このブロックは、スクリプトが直接実行された場合にのみ実行されます。
if __name__ == '__main__':

    input_directory = 'results/gemini' 
    
    # Visualizerのインスタンスを作成します。
    visualizer_1 = Visualizer()
    visualizer_2 = Visualizer()

    #results_load("results/gemini",visualizer_1)
    results_load("results/TSMMG",visualizer_1)
    #results_load("results/BioT5",visualizer_2)

    #visualizer_1.plot_similarity(visualizer_2)
    visualizer_1.plot_score_shift(5)
    visualizer_1.visualize_crossover()


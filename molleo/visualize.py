import yaml
import glob
import os
import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw
from tdc import Evaluator

from mol_data import Mol_Data

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
        self.smi2mlc = {}
        self.num_generation = 0

        self.diversity_evaluator = Evaluator(name = 'Diversity')

    def results_load(self, input_dir):
        # 指定されたディレクトリ内で'population_'で始まるすべてのYAMLファイルを検索します。
        yaml_files = glob.glob(os.path.join(input_dir, 'population_*.yaml'))
        self.num_generation = len(yaml_files)

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
                self.populations.append(population)

                for mlc in population:
                    if mlc.smi not in self.smi2mlc:
                        self.smi2mlc[mlc.smi] = mlc
            
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

                    family = [offspring.mol, parent1.mol, parent2.mol]
                    legends = [f"parent1 score : {parent1.score:.3f}",
                                f"parent2 score : {parent2.score:.3f}",
                                f"offspring score : {offspring.score:.3f}"]

                    img = Draw.MolsToGridImage(family, molsPerRow=3, subImgSize=(300, 300), legends=legends)
                    
                    output_path = os.path.join(output_gene_dir,f"{offspring.smi}.png")
                    img.save(output_path)
            num_G += 1



# このブロックは、スクリプトが直接実行された場合にのみ実行されます。
if __name__ == '__main__':

    input_directory = 'results' 
    
    # Visualizerのインスタンスを作成します。
    visualizer = Visualizer()
    
    visualizer.results_load(input_directory)
    visualizer.plot_score_shift(10)
    visualizer.plot_diversity_shift()
    #visualizer.visualize_crossover()


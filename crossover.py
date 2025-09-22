import random

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

# RDKitのエラーログを無効化し、コンソールの出力をクリーンに保つ
rdBase.DisableLog('rdApp.error')


def cut(mol):
    """
    分子内の環に属さない単結合をランダムに1つ選び、その位置で分子を切断します。
    切断された箇所にはダミー原子が付加され、2つのフラグメント（分子断片）が生成されます。

    Args:
        mol (Mol): 切断対象のRDKit Molオブジェクト。

    Returns:
        tuple or None:
            - 成功した場合: 2つのフラグメント（Molオブジェクト）を含むタプル。
            - 失敗した場合（切断可能な結合がないなど）: None。
    """
    # 分子内に環外の単結合 '[*]-;!@[*]' が存在するかチェック
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
        return None

    # 環外の単結合をランダムに一つ選択
    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')))
    # 選択した原子ペア間の結合のインデックスを取得
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    # 指定した結合インデックスで分子をフラグメント化。切断点にダミー原子(ラベル=1)を付加。
    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        # フラグメント化された分子を個別のMolオブジェクトに分割し、サニタイズ（妥当性検証）する
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        # サニタイズに失敗した場合はNoneを返す
        return None


def cut_ring(mol):
    """
    分子内の環を2箇所で切断し、環を開裂させて2つのフラグメントを生成します。
    切断パターンは、隣接しない2つの結合を切るか、隣接する2つの結合を切る（1原子を取り除く）かの
    いずれかがランダムに選択されます。最大10回試行します。

    Args:
        mol (Mol): 切断対象のRDKit Molオブジェクト。

    Returns:
        tuple or None:
            - 成功した場合: 2つのフラグメント（Molオブジェクト）を含むタプル。
            - 失敗した場合（適切な環構造がない、フラグメント化に失敗など）: None。
    """
    # 有効なフラグメントが得られるまで最大10回試行
    for i in range(10):
        # 50%の確率で、4原子からなる環構造の一部を切断
        if random.random() < 0.5:
            # 4原子が連続する環構造 '[R]@[R]@[R]@[R]' が存在するかチェック
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')):
                return None
            # 該当箇所をランダムに選択
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')))
            # 1番目と2番目、3番目と4番目の原子間の結合を切断対象とする
            bis = ((bis[0], bis[1]), (bis[2], bis[3]),)
        # 残り50%の確率で、次数が2より大きい環原子を含む部分を切断
        else:
            # '[R]@[R;!D2]@[R]' (分岐点を持つ3原子環構造) が存在するかチェック
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')):
                return None
            # 該当箇所をランダムに選択
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')))
            # 1番目と2番目、2番目と3番目の原子間の結合を切断対象とする
            bis = ((bis[0], bis[1]), (bis[1], bis[2]),)

        # 選択された原子ペア間の結合インデックスを取得
        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

        # 指定した結合で分子をフラグメント化
        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)])

        try:
            # フラグメントを個別のMolオブジェクトに分割
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            # ちょうど2つのフラグメントが生成された場合のみ成功とみなし、それを返す
            if len(fragments) == 2:
                return fragments
        except ValueError:
            # サニタイズに失敗した場合は次の試行へ
            continue
    
    # 10回試行しても成功しなかった場合はNoneを返す
    return None


def ring_OK(mol):
    """
    分子内の環構造が化学的に妥当であるかをチェックします。
    具体的には、環状アレン、7員環以上の大環状構造、3員環または4員環内の二重結合など、
    不安定、または合成困難な構造が含まれていないかを確認します。

    Args:
        mol (Mol): チェック対象のRDKit Molオブジェクト。

    Returns:
        bool: 環構造が妥当であればTrue、そうでなければFalse。
    """
    # 環が存在しない場合は常にTrue
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]')):
        return True

    # 環状アレン '[R]=[R]=[R]' の存在をチェック
    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts('[R]=[R]=[R]'))

    # 分子内のすべての環情報を取得
    cycle_list = mol.GetRingInfo().AtomRings()
    # 最大の環サイズを計算
    max_cycle_length = max([len(j) for j in cycle_list]) if cycle_list else 0
    # 7員環以上を大環状化合物（マクロサイクル）と判定
    macro_cycle = max_cycle_length > 6

    # 3員環または4員環内に二重結合 '[r3,r4]=[r3,r4]' が存在するかチェック
    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]'))

    # 上記のいずれかの不安定構造を含まない場合にのみTrueを返す
    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


# TODO: データセットから計算するか、mainから設定するようにする
# 分子サイズの平均と標準偏差（生成される分子のサイズを制限するために使用）
average_size = 39.15
size_stdev = 3.50


def mol_ok(mol):
    """
    生成された分子が全体として妥当であるかをチェックします。
    1. RDKitによってサニタイズ（化学構造の妥当性検証）可能か。
    2. 分子の原子数が妥当な範囲内（5原子以上かつ、目標サイズ範囲内）にあるか。

    Args:
        mol (Mol): チェック対象のRDKit Molオブジェクト。

    Returns:
        bool: 分子が妥当であればTrue、そうでなければFalse。
    """
    try:
        # 分子のサニタイズを試みる
        Chem.SanitizeMol(mol)
        # 正規分布に従う目標分子サイズを生成
        target_size = size_stdev * np.random.randn() + average_size
        # 原子数が5より大きく、かつ目標サイズより小さいかチェック
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except ValueError:
        # サニタイズに失敗した場合はFalseを返す
        return False


def crossover_ring(parent_A, parent_B):
    """
    2つの親分子間で環を含むフラグメントを交換することにより、新しい子分子を生成（交叉）します。
    両親から`cut_ring`でフラグメントを生成し、それらを組み合わせて新しい分子を再構築します。
    最大10回試行します。

    Args:
        parent_A (Mol): 1番目の親分子。
        parent_B (Mol): 2番目の親分子。

    Returns:
        Mol or None:
            - 成功した場合: 生成された新しい子分子（Molオブジェクト）。
            - 失敗した場合: None。
    """
    # 環を持つ分子のSMARTS
    ring_smarts = Chem.MolFromSmarts('[R]')
    # 両親ともに環を持たない場合は交叉不可
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    # フラグメントを結合するためのSMARTS反応テンプレート
    # '[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]' : 2つのフラグメントを単結合で繋ぐ
    # '([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]' : 環を閉じるように結合する
    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']

    # 有効な子分子が生成されるまで最大10回試行
    for i in range(10):
        # 両親から環フラグメントを切り出す
        fragments_A = cut_ring(parent_A)
        fragments_B = cut_ring(parent_B)

        # どちらかのフラグメント化が失敗したら交叉不可
        if fragments_A is None or fragments_B is None:
            return None

        # フラグメントの組み合わせを試す
        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            # 親Aの各フラグメントと親Bの各フラグメントを総当たりで結合
            for fa in fragments_A:
                for fb in fragments_B:
                    prod = rxn1.RunReactants((fa, fb))
                    if len(prod) > 0:
                        new_mol_trial.append(prod[0])

        # 生成された中間生成物からさらに環を閉じる反応を試す
        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_ok(m): # 中間生成物が妥当かチェック
                    new_mols += list(rxn2.RunReactants((m,)))

        # 最終的に生成された分子の妥当性をチェック
        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if mol_ok(m) and ring_OK(m): # 分子全体と環構造が妥当かチェック
                new_mols2.append(m)

        # 妥当な分子が1つでもあれば、その中からランダムに1つを返す
        if len(new_mols2) > 0:
            return random.choice(new_mols2)

    # 10回試行しても成功しなかった場合はNoneを返す
    return None


def crossover_non_ring(parent_A, parent_B):
    """
    2つの親分子間で環を含まないフラグメントを交換することにより、新しい子分子を生成（交叉）します。
    両親から`cut`でフラグメントを生成し、それらを組み合わせて新しい分子を再構築します。
    最大10回試行します。

    Args:
        parent_A (Mol): 1番目の親分子。
        parent_B (Mol): 2番目の親分子。

    Returns:
        Mol or None:
            - 成功した場合: 生成された新しい子分子（Molオブジェクト）。
            - 失敗した場合: None。
    """
    # 有効な子分子が生成されるまで最大10回試行
    for i in range(10):
        # 両親から非環状フラグメントを切り出す
        fragments_A = cut(parent_A)
        fragments_B = cut(parent_B)
        
        # どちらかのフラグメント化が失敗したら交叉不可
        if fragments_A is None or fragments_B is None:
            return None
        
        # 2つのフラグメントを単結合で繋ぐためのSMARTS反応
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        
        new_mol_trial = []
        # 親Aの各フラグメントと親Bの各フラグメントを総当たりで結合
        for fa in fragments_A:
            for fb in fragments_B:
                prod = rxn.RunReactants((fa, fb))
                if len(prod) > 0:
                    new_mol_trial.append(prod[0])

        # 生成された分子の妥当性をチェック
        new_mols = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_ok(mol): # 分子が妥当かチェック
                new_mols.append(mol)

        # 妥当な分子が1つでもあれば、その中からランダムに1つを返す
        if len(new_mols) > 0:
            return random.choice(new_mols)

    # 10回試行しても成功しなかった場合はNoneを返す
    return None


def crossover(parent_A, parent_B):
    """
    2つの親分子を受け取り、遺伝的アルゴリズムの交叉操作を実行します。
    50%の確率で環部分の交叉(`crossover_ring`)を、50%の確率で非環部分の交叉(`crossover_non_ring`)を
    試みます。生成された子分子が親と異なる場合に、その子分子を返します。

    Args:
        parent_A (Mol): 1番目の親分子。
        parent_B (Mol): 2番目の親分子。

    Returns:
        Mol or None:
            - 成功した場合: 生成された新しい子分子（Molオブジェクト）。
            - 失敗した場合: None。
    """
    # 親分子のSMILESをリストに保存（後で子との重複をチェックするため）
    parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    
    # 親分子をケクレ化（芳香環の二重結合を明示的に表現）する
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)
    except ValueError:
        # ケクレ化に失敗しても処理を続行
        pass

    # 有効な子分子が生成されるまで最大10回試行
    for i in range(10):
        # 50%の確率で非環状交叉を選択
        if random.random() <= 0.5:
            new_mol = crossover_non_ring(parent_A, parent_B)
            # 交叉が成功した場合
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                # 生成された子が有効で、かつ親と重複しないことを確認
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol
        # 残り50%の確率で環状交叉を選択
        else:
            new_mol = crossover_ring(parent_A, parent_B)
            # 交叉が成功した場合
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                # 生成された子が有効で、かつ親と重複しないことを確認
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol

    # 10回試行しても有効な子を生成できなかった場合はNoneを返す
    return None
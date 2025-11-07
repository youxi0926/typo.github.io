import os
import re
import Levenshtein
from pyxdameraulevenshtein import damerau_levenshtein_distance
import pandas as pd
from difflib import SequenceMatcher
import difflib
from collections import defaultdict, Counter
import json
from typing import Dict, Tuple, Any, List, Set

# ====================================================================-
# --------タイポドメイン抽出----------

# ドメイン部抽出
def extract_domain(email):
    return email.split('@')[1] if '@' in email else ''

# 2つの文字列間の異なる部分を抽出
def get_mismatched_part(correct, input_):
    matcher = SequenceMatcher(None, correct, input_)
    mismatched = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            mismatched.append(input_[j1:j2])
    return ''.join(mismatched)

def filter_domain_differences_with_mismatch(input_path, output_path, threshold=5):     # タイポデータの抽出と整形
    df = pd.read_csv(input_path)  # CSV読み込み

    # ドメイン部の抽出
    df["correct_domain"] = df["correct_address"].astype(str).apply(extract_domain)
    df["input_domain"] = df["input_address"].astype(str).apply(extract_domain)

    # Damerau-Levenshtein距離を計算
    df["domain_edit_distance"] = df.apply(
        lambda row: damerau_levenshtein_distance(row["correct_domain"], row["input_domain"]),
        axis=1
    )

    # ドメイン部が異なる（距離 > 0 かつ <= threshold）のみ抽出
    filtered_df = df[
        (df["domain_edit_distance"] > 0) &
        (df["domain_edit_distance"] <= threshold)
    ].reset_index(drop=True)

    # 差分部分（mismatched_part）の抽出
    filtered_df["mismatched_part"] = filtered_df.apply(
        lambda row: get_mismatched_part(row["correct_domain"], row["input_domain"]),
        axis=1
    )

    # edit_distanceを整形して出力
    filtered_df["edit_distance"] = filtered_df["domain_edit_distance"]
    final_df = filtered_df[["user_id", "step_id", "correct_address", "input_address", "edit_distance", "mismatched_part"]]

    # 保存
    final_df.to_csv(output_path, index=False)
    print(f"[INFO] ドメインの違いと差分を出力しました: {output_path}")

# 差分4での抽出例
if __name__ == "__main__":
    filter_domain_differences_with_mismatch(
        input_path="filtered_address.csv",
        output_path="filtered_domain_typos_dl4.csv",
        threshold=4
    )

# ====================================================================
# --------原因別分類-----------

keyboard_adjacent = { # キーボード隣接キーマップ
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qws', 's': 'qwedazx', 'd': 'erfcsx', 'f': 'rtdgvcj',
    'g': 'tyfhvbn', 'h': 'yugjnb', 'j': 'uikhmnf', 'k': 'ijolm', 'l': 'okp',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk,',
    '.': ',/', ',': 'm.',
    '-': '^'
}

symmetric_key_pairs = [('f', 'j'), ('d', 'k'), ('s', 'l'), ('a', ';')] # 対称配置キー誤打（例: f ↔ j）
homoglyph_pairs = [('1', 'l'), ('0', 'o'), ('i', 'l'), ('rn', 'm'), ('а', 'a'), ('b', 'd')]  # # ホモグラフ, キリル文字の'a'など

def keyboard_adjacent_check(c1, c2):
    return c1.lower() in keyboard_adjacent and c2.lower() in keyboard_adjacent[c1.lower()]

def is_symmetric_mismatch(c1, c2): #対称キーの誤打であるか(一文字ずつ判定)
    for a, b in symmetric_key_pairs:
        if (c1 == a and c2 == b) or (c1 == b and c2 == a):
            return True
    return False

def is_visual_homoglyph(c1, c2): #視覚類似文字（ホモグリフ）の誤打であるか(一文字ずつ判定)
    for a, b in homoglyph_pairs:
        if (c1 == a and c2 == b) or (c1 == b and c2 == a):
            return True
    return False

def is_valid_tld(tld): #TLDとして妥当か(長さを2〜4文字に限定し、TLDの制約に基づいて判定)
    return len(tld) in [2, 3, 4] and tld.isalpha()


# --------------原因別集計-----------------
# 原因別分類関数
def classify_edit_ops_japanese(correct, typo):
    correct_parts = []
    typo_parts = []
    causes = set()

    if damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1:
        causes.add("入力順序ミス")

    ops = Levenshtein.editops(correct, typo)
    for tag, src_i, tgt_i in ops:
        c1 = correct[src_i] if src_i < len(correct) else ''
        c2 = typo[tgt_i] if tgt_i < len(typo) else ''

        if tag == 'replace':
            correct_parts.append(c1)
            typo_parts.append(c2)
            if keyboard_adjacent_check(c1, c2):
                causes.add("隣接キー誤打")
            elif is_symmetric_mismatch(c1, c2):
                causes.add("左右対称キー誤打")
            elif is_visual_homoglyph(c1, c2):
                causes.add("ホモグリフ（視覚類似文字）")
            else:
                causes.add("スペルミス（認知ミス）")

        elif tag == 'insert':
            correct_parts.append('')
            typo_parts.append(c2)   
            causes.add("二重入力")

        elif tag == 'delete':
            correct_parts.append(c1)
            typo_parts.append('')
            if c1 == '.':
                causes.add("ドット抜け")
            else:
                causes.add("入力漏れ")

    # TLD違い
    correct_tld = correct.split('.')[-1]
    typo_tld = typo.split('.')[-1] if '.' in typo else ''
    if typo_tld and typo_tld != correct_tld and is_valid_tld(typo_tld):
        if keyboard_adjacent_check(correct_tld, typo_tld):
            causes.add("隣接キー誤打")

    return {
        "cause": '・'.join(sorted(causes)),
        "correct_part": ' '.join(correct_parts),
        "mismatched_part": ' '.join(typo_parts)
    }

def get_transposed_pair(correct, typo):
    # 入力順序ミスが単独で発生している場合、入れ替わった文字ペアを返す
    # Damerau-Levenshtein距離が1の単独転置の場合にのみ有効 
    if damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1:
        for i in range(len(correct) - 1):
            if correct[i] == typo[i+1] and correct[i+1] == typo[i] and correct[:i] == typo[:i] and correct[i+2:] == typo[i+2:]:
                # 例: 'ab' -> 'ba'
                return (correct[i], correct[i+1])
    return None

# --------------------------------------------------------------------------
# ⭐ 欠落していた識別関数を追加
# --------------------------------------------------------------------------

def identify_single_replacement(correct, typo):
    """DL距離1の置換・挿入・削除ミスの差分文字ペアを特定する（予測生成用）。"""
    matcher = SequenceMatcher(None, correct, typo)
    ops = matcher.get_opcodes()
    
    replacements = [(correct[i1:i2], typo[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag == 'replace']
    inserts = [(typo[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag == 'insert']
    deletes = [(correct[i1:i2]) for tag, i1, i2, j1, j2 in ops if tag == 'delete']
    
    # 単一のミスであるかチェック（複雑な複合ミスは予測生成時には無視する）
    if len(replacements) == 1 and not inserts and not deletes:
        c1 = replacements[0][0]
        c2 = replacements[0][1]
        if len(c1) == 1 and len(c2) == 1:
            return (c1, c2)  # 置換
            
    if len(inserts) == 1 and not replacements and not deletes:
        return ('（空）', inserts[0][0]) # 挿入 (二重入力)

    if len(deletes) == 1 and not replacements and not inserts:
        return (deletes[0][0], '（空）') # 削除 (入力漏れ/ドット抜け)

    return ('', '') # 複合または非距離1ミス


def analyze_for_ranking(csv_path):
    """CSVを読み込み、ランキング用の個別ミス重み (W_individual) を計算する。"""
    df = pd.read_csv(csv_path)
    
    individual_rank_weights = defaultdict(dict)
    all_causes = []
    cause_diff_counter = defaultdict(Counter)

    for _, row in df.iterrows():
        correct = extract_domain(str(row['correct_address']))
        typo = extract_domain(str(row['input_address']))
        cause_field = str(row['cause'])
        causes = [c.strip() for c in cause_field.split('・')]

        diffs = extract_ngram_diffs(correct, typo)
        all_causes.extend(causes)

        for cause in causes:
            for c1, c2 in diffs:
                # 差分パターンをそのまま集計
                cause_diff_counter[cause][(c1, c2)] += 1

    # 大分類の割合計算 (レポート用)
    cause_counts = Counter(all_causes)
    total_major_events = sum(cause_counts.values())
    major_ratios = {k: round(v / total_major_events, 3) for k, v in cause_counts.items()}

    # W_individual (個別ミス件数 / 全タイポイベント総数) の計算
    total_typo_events = sum(sum(counter.values()) for counter in cause_diff_counter.values())
    
    for cause, counter in cause_diff_counter.items():
        for key, count in counter.items():
            # W_individual = 個別ミス件数 / 全タイポイベント総数
            rank_score = count / total_typo_events
            individual_rank_weights[cause][key] = rank_score
            
    # レポート出力 (ここでは省略しますが、既存のanalyze_ngram_differencesの出力ロジックをここに統合する必要があります)
    
    return major_ratios, individual_rank_weights

#--------------------------------------------------------------------------------------
# cause, correctの付与csvファイル出力関数
def append_typo_causes(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    causes = []
    correct_parts = []
    typo_parts = []

    for _, row in df.iterrows():
        correct = extract_domain(str(row['correct_address']))
        typo = extract_domain(str(row['input_address']))
        result = classify_edit_ops_japanese(correct, typo)
        causes.append(result["cause"])
        correct_parts.append(result["correct_part"])
        typo_parts.append(result["mismatched_part"])

    df['cause'] = causes
    df['correct_part'] = correct_parts
    df['mismatched_part'] = typo_parts

    # 列の順序を変更
    desired_columns = [
        'user_id', 'step_id',
        'correct_address', 'input_address',
        'edit_distance',
        'correct_part', 'mismatched_part', 'cause'
    ]

    # 対象の列があるものだけに限定
    df = df[[col for col in desired_columns if col in df.columns]]

    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    # print(f"→ '{output_csv_path}' に出力しました。") ##→ 'domaintypos_dl4_causes2.csv' に出力しました。

# 'correct_part'を付けて、"domaintypos_dl4_causes2.csv"に出力
append_typo_causes("filtered_domain_typos_dl4.csv", "domaintypos_dl4_causes2.csv")

#--------------------------------------------------------------------------------------
# 差分抽出（difflibベース、n文字対応）
def extract_ngram_diffs(correct, typo):
    sm = difflib.SequenceMatcher(None, correct, typo)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        src = correct[i1:i2] or '（空）'
        tgt = typo[j1:j2] or '（空）'
        diffs.append((src, tgt))
    return diffs


# 原因ごとの差分集計
def analyze_ngram_differences(csv_path):
    df = pd.read_csv(csv_path)
    cause_diff_counter = defaultdict(Counter)

    for _, row in df.iterrows():
        correct = extract_domain(str(row['correct_address']))
        typo = extract_domain(str(row['input_address']))
        cause_field = str(row['cause'])
        causes = [c.strip() for c in cause_field.split('・')]

# 入力順序ミスの場合は特殊な処理を行う
        if "入力順序ミス" in causes:
            transposed_pair = get_transposed_pair(correct, typo)
            if transposed_pair:
                c1, c2 = transposed_pair
                # 入れ替わったペアをタグとして集計（例: 'e r' -> 1件）
                cause_diff_counter["入力順序ミス"][f'{c1} {c2} -> {c2} {c1}'] += 1
                
                # 転置ミスは他の差分パターン分析から除外して重複を防ぐため、次の行へスキップ
                continue

        diffs = extract_ngram_diffs(correct, typo)

        for cause in causes:
            # 入力順序ミスの場合は既に集計済みのため、スキップ
            if cause == "入力順序ミス":
                continue 
            
            for c1, c2 in diffs:
                
                # difflibの差分タグに基づき、原因と差分パターンが一致するか検証する

                is_replace = (c1 != '（空）' and c2 != '（空）')
                is_insert = (c1 == '（空）' and c2 != '（空）')
                is_delete = (c1 != '（空）' and c2 == '（空）')

                # --------------------------------------------------------------------------
                # 1. 物理的/認知的な置換ミス (Replace)
                # --------------------------------------------------------------------------
                if is_replace:
                    if cause == "隣接キー誤打" and keyboard_adjacent_check(c1, c2):
                        cause_diff_counter[cause][(c1, c2)] += 1
                    elif cause == "ホモグリフ（視覚類似文字）" and is_visual_homoglyph(c1, c2):
                        cause_diff_counter[cause][(c1, c2)] += 1
                    elif cause == "左右対称キー誤打" and is_symmetric_mismatch(c1, c2):
                        cause_diff_counter[cause][(c1, c2)] += 1
                    elif cause == "スペルミス（認知ミス）":
                        # 上記の物理的・視覚的なミスに該当しない置換は、スペルミスとしてカウント
                        # ※ TLD置換（例: com→cojp）で別TLD結合タグが付いていないものも含む
                        cause_diff_counter[cause][(c1, c2)] += 1
                
                # --------------------------------------------------------------------------
                # 2. 構造的なミス (Insert / Delete)
                # --------------------------------------------------------------------------
                elif is_insert:
                    if cause == "二重入力":
                        cause_diff_counter[cause][(c1, c2)] += 1

                elif is_delete:
                    if cause == "入力漏れ" and c1 != '.':
                        # ドット以外の削除
                        cause_diff_counter[cause][(c1, c2)] += 1
                    elif cause == "ドット抜け" and c1 == '.':
                        # ドットの削除
                        cause_diff_counter[cause][(c1, c2)] += 1
                        
                # --------------------------------------------------------------------------
                # 3. 複雑なミス (順序入れ替え)
                # --------------------------------------------------------------------------
                elif cause == "入力順序ミス":
                    # difflibが転置を削除/挿入として扱うため、ここではタグの存在のみでカウントし、
                    # 差分パターンをそのまま集計。ただし、前述の delete/insert に重複する可能性があるため、
                    # 後の詳細分析のためのデータとして、一旦そのまま集計する（現状維持）。
                    cause_diff_counter[cause][(c1, c2)] += 1



    # 原因別件数出力
    for cause, counter in cause_diff_counter.items():
        print(f"\n【原因: {cause}】")

        # フルver
        # for (c1, c2), count in counter.most_common():
        #     print(f"  {c1} → {c2:<10}: {count}件")

        
        # 入力順序ミスの場合、キーは文字列なので、一つの変数(key_pair)で受け取る
        if cause == "入力順序ミス":
            for key_pair, count in counter.most_common(): # key_pair は 'e r -> r e'
                print(f"  {key_pair:<10}: {count}件")
        
        # それ以外の原因の場合、キーは(c1, c2)のタプルなので、(c1, c2)で受け取る
        else:
            for (c1, c2), count in counter.most_common(): 
                print(f"  {c1} → {c2:<10}: {count}件")
        

# 実行
analyze_ngram_differences("domaintypos_dl4_causes2.csv")

#---------タイポ原因別集計と割合（重み）--------
df = pd.read_csv("domaintypos_dl4_causes2.csv")  # CSV読み込み

# cause列から個別原因を抽出・集計
all_causes = []
for cause in df["cause"].dropna():
    causes = [c.strip() for c in cause.split("・")]
    all_causes.extend(causes)

# 集計と割合計算
cause_counts = Counter(all_causes)
total = sum(cause_counts.values())
cause_ratios = {k: round(v / total, 3) for k, v in cause_counts.items()}

# 表示を整形して出力
print("==============================================================================")
print("■ 原因別集計と割合（重み）:\n")
for cause in sorted(cause_counts, key=cause_counts.get, reverse=True):
    count = cause_counts[cause]
    ratio = cause_ratios[cause]
    print(f"{cause:<25}: {count:>5}件 ({ratio:>5.3f})")


# ===================================================
# --------ドメインランキング----------

def typo_domain_ranking_with_reason_jp(input_path, correct_domain, max_distance=4): #####　←←←←←←←←←←←←←←←←←←←←←←←←←DL距離指定
    df = pd.read_csv(input_path)
    df["input_domain"] = df["input_address"].astype(str).apply(extract_domain)

    typo_df = df[df["input_domain"].apply(lambda d: damerau_levenshtein_distance(correct_domain, d) <= max_distance)].copy()
    typo_df["distance"] = typo_df["input_domain"].apply(lambda d: damerau_levenshtein_distance(correct_domain, d))
    typo_df = typo_df[typo_df["input_domain"] != correct_domain]

    grouped = typo_df.groupby("input_domain").agg(
        count=("input_domain", "count"),
        distance=("distance", "first")
    ).reset_index()

    total_typos = grouped["count"].sum()
    grouped["percentage"] = grouped["count"] / total_typos * 100
    grouped["cause"] = grouped["input_domain"].apply(lambda typo: classify_edit_ops_japanese(correct_domain, typo))

    grouped = grouped.sort_values(by=["count", "distance"], ascending=[False, True]).reset_index(drop=True)

    print(f"\n '{correct_domain}' に対するタイポドメインランキング（DL距離 ≦ {max_distance}）:\n")
    for i, row in grouped.iterrows():
        print(f"{i+1}位　{row['input_domain']}（{row['count']}回, 距離: {row['distance']}, 割合: {row['percentage']:.1f}%, 原因: {row['cause']}）")
    print()

if __name__ == "__main__":
    correct_domain = input("\n入力されたドメインのタイポランキングを表示（例: treasurefactory.co.jp）: ").strip()
    typo_domain_ranking_with_reason_jp("filtered_domain_typos_dl4.csv", correct_domain)


# ===============================================
# タイポドメイン生成関数

typo_weights = cause_ratios

# ===================================================================
# --------予測型タイポドメイン生成関数----------
# ===================================================================

# homoglyphs は 'rn' : ['m'] のような生成用辞書に置き換え
# (元のコードには homoglyphs が定義されていなかったため、仮定して拡張します)
HOMOGLYPHS_FOR_GENERATOR = {'1': ['l'], '0': ['o'], 'i': ['l'], 'l': ['i'], 'r': ['m'], 'b': ['d'], 'd': ['b']} 
TLD_ALTERNATIVES = {'com': ['co.jp', 'net'], 'co.jp': ['com', 'co'], 'net': ['com', 'co.jp']} # tld例

symmetric_key_pairs = [('f', 'j'), ('d', 'k'), ('s', 'l'), ('a', ';')] # 対称配置キー誤打（例: f ↔ j）
homoglyph_pairs = [('1', 'l'), ('0', 'o'), ('i', 'l'), ('rn', 'm'), ('а', 'a'), ('b', 'd')]  # # ホモグラフ, キリル文字の'a'など


# def typo_generator_ranked(domain: str, top_n: int = 30):
# typo_generator_ranked 関数の定義行を探して修正
def typo_generator_ranked(domain: str, individual_weights: dict, top_n: int = 30):
    # ... (関数の本体はそのまま) ...
    """
    タイポ原因の重み付けに基づき、発生可能性の高いタイポドメインを生成・ランキングする。
    """
    variants = defaultdict(lambda: (set(), 0)) # {typo_domain: (causes_set, score)}

    # ドメイン全体に対して処理を適用
    for i in range(len(domain)):
        c = domain[i]
        
        # --- 基本的な距離1のミス ---

        # 1. 入力漏れ (Deletion)
        typo = domain[:i] + domain[i+1:]
        variants[typo] = variants.get(typo, (set(), 0))
        variants[typo][0].add("入力漏れ")

        # 2. 二重入力 (Insertion/Repetition)
        typo = domain[:i] + c + c + domain[i+1:]
        variants[typo] = variants.get(typo, (set(), 0))
        variants[typo][0].add("二重入力")

        # 3. 隣接キー誤打 (Substitution)
        for adj in keyboard_adjacent.get(c.lower(), ''):
            typo = domain[:i] + adj + domain[i+1:]
            variants[typo] = variants.get(typo, (set(), 0))
            variants[typo][0].add("隣接キー誤打")

        # 4. ホモグリフ・視覚類似文字 (Substitution)
        for g in HOMOGLYPHS_FOR_GENERATOR.get(c.lower(), []):
            typo = domain[:i] + g + domain[i+1:]
            variants[typo] = variants.get(typo, (set(), 0))
            variants[typo][0].add("ホモグリフ（視覚類似文字）")
            
        # 5. 左右対称キー誤打 (Substitution)
        for a, b in symmetric_key_pairs:
            if c == a: 
                typo = domain[:i] + b + domain[i+1:]
                variants[typo] = variants.get(typo, (set(), 0))
                variants[typo][0].add("左右対称キー誤打")
            elif c == b:
                typo = domain[:i] + a + domain[i+1:]
                variants[typo] = variants.get(typo, (set(), 0))
                variants[typo][0].add("左右対称キー誤打")

    # --- 構造的なミス ---

    # 6. 入力順序ミス (Transposition)
    for i in range(len(domain) - 1):
        swapped = domain[:i] + domain[i+1] + domain[i] + domain[i+2:]
        variants[swapped] = variants.get(swapped, (set(), 0))
        variants[swapped][0].add("入力順序ミス")

    # 8. ドット抜け (Deletion of dot)
    if '.' in domain:
        dotless = domain.replace('.', '', 1) # 最初に見つかったドットのみ削除
        variants[dotless] = variants.get(dotless, (set(), 0))
        variants[dotless][0].add("ドット抜け")

    # --- 統合とスコア集計 ---

    ranked_results = []
    
    for typo, (causes, _) in variants.items():
        if typo == domain: continue
        
        final_score = 0
        
        # 距離1のミスの文字ペアを特定
        c1, c2 = identify_single_replacement(domain, typo) 
        
        # 1. 個別ミス重み (W_individual) の適用
        for cause in causes:
            W_individual = 0.0 # 見つからなかった場合のデフォルト値
            
            # 内部重み (W_individual) の参照ロジック
            # 転置ミスは、キーが文字列のまま参照される必要があります。
            if cause == "入力順序ミス":
                 # キーは文字列 'e r -> r e'
                 transposed_pair = get_transposed_pair(domain, typo)
                 if transposed_pair:
                    k1, k2 = transposed_pair
                    key = f'{k1} {k2} -> {k2} {k1}'
                    W_individual = individual_weights.get(cause, {}).get(key, 0.0)
            
            elif cause in {"隣接キー誤打", "ホモグリフ（視覚類似文字）", "左右対称キー誤打", "スペルミス（認知ミス）"}:
                # 置換系 (c1, c2)
                key = (c1, c2)
                W_individual = individual_weights.get(cause, {}).get(key, W_individual)
                if W_individual == 0.0: # 逆順もチェック (置換のみ)
                    W_individual = individual_weights.get(cause, {}).get((c2, c1), 0.0)
            
            elif cause in {"入力漏れ", "ドット抜け"}:
                # 削除系 (c1, '（空）')
                key = (c1, '（空）')
                W_individual = individual_weights.get(cause, {}).get(key, 0.0)
            
            elif cause == "二重入力":
                # 挿入系 ('（空）', c2)
                key = ('（空）', c2) 
                W_individual = individual_weights.get(cause, {}).get(key, 0.0)

            # 最終スコア: W_individualを単純に加算 (ボーナス/ペナルティはここでは適用しない)
            final_score += W_individual

        # 2. 複合ミス ペナルティの適用 (W_individualの合計が過大になるのを防ぐ)
        if len(causes) > 1:
            final_score *= 0.5 

        
        # Damerau-Levenshtein距離も併せて計算 (分析結果の検証に役立つ)
        distance = damerau_levenshtein_distance(domain, typo)

        ranked_results.append({
            "typo": typo,
            "causes": '・'.join(sorted(causes)),
            "score": round(final_score, 3), # ここを final_score に修正
            "distance": distance
        })

    # スコア降順、距離昇順でランキング
    ranked_results.sort(key=lambda x: (x['score'], -x['distance']), reverse=True)

    return ranked_results[:top_n]


# ===================================================================
# --------実行部分----------
if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # 必須ファイルパス設定
    INPUT_FILE = "filtered_address.csv"
    DL4_FILTERED_FILE = "filtered_domain_typos_dl4.csv"
    CAUSES_CSV_FILE = "domaintypos_dl4_causes2.csv"
    DL_THRESHOLD = 4

    # 1. タイポの抽出とフィルタリング (中間ファイル生成)
    filter_domain_differences_with_mismatch(INPUT_FILE, DL4_FILTERED_FILE, DL_THRESHOLD)

    # 2. 原因分類を付与しCSVに出力
    append_typo_causes(DL4_FILTERED_FILE, CAUSES_CSV_FILE)
    
    # --------------------------------------------------------------------------
    # 3. 分析の実行: 個別ミスの重み (W_individual) を計算
    # --------------------------------------------------------------------------
    
    # analyze_for_rankingを実行し、大分類の重みと個別ミスの重みの両方を受け取る
    major_weights, individual_rank_weights = analyze_for_ranking(CAUSES_CSV_FILE)


    # --------------------------------------------------------------------------
    # 4. ドメインランキング生成の実行
    # --------------------------------------------------------------------------
    
    correct_domain = input("\n 入力されたドメインのタイポドメイン候補を生成する（例: treasurefactory.co.jp）: ").strip()
    
    if major_weights and individual_rank_weights:
        print("\n" + "=" * 78)
        print(f"'{correct_domain}' に対する予測タイポドメインランキング:\n")
        
        # typo_generator_ranked 関数に、計算した個別ミス重み (individual_rank_weights) を渡す
        # すべての引数をキーワードで指定し、順序依存性をなくす（より安全）
        predicted_typos = typo_generator_ranked(
            domain=correct_domain,
            individual_weights=individual_rank_weights, # 新しいスコアの主軸となる個別重み
            top_n=10
        )
        
        for i, r in enumerate(predicted_typos):
            print(f"{i+1}位 {r['typo']:<30} (スコア: {r['score']:.5f}, 距離: {r['distance']}, 原因: {r['causes']})")
    else:
        print("\n[エラー] 重みデータが計算されていないため、ランキング生成を実行できませんでした。")

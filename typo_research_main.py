import os
import re
import Levenshtein
from pyxdameraulevenshtein import damerau_levenshtein_distance
import pandas as pd
from difflib import SequenceMatcher
import difflib
from collections import defaultdict, Counter
import json

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

def filter_domain_differences_with_mismatch(input_path, output_path, threshold=5):
    df = pd.read_csv(input_path) # CSV読み込み

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
    correct_domain = input("\n正しいドメイン名を入力してください（例: treasurefactory.co.jp）: ").strip()
    typo_domain_ranking_with_reason_jp("filtered_domain_typos_dl4.csv", correct_domain)

# ===========================================================================================

keyboard_adjacent = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qws', 's': 'qwedazx', 'd': 'erfcsx', 'f': 'rtdgvcj',
    'g': 'tyfhvbn', 'h': 'yugjnb', 'j': 'uikhmnf', 'k': 'ijolm', 'l': 'okp',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk,',
    '.': ',/', ',': 'm.'
}

symmetric_key_pairs = [('f', 'j'), ('d', 'k'), ('s', 'l')] # 対称配置キー誤打（例: f ↔ j）
homoglyph_pairs = [('1', 'l'), ('0', 'o'), ('i', 'l'), ('rn', 'm'), ('а', 'a'), ('b', 'd')]  # ホモグラフ, キリル文字の'a'など
# homoglyphs = {'1': ['l'], '0': ['o'], 'i': ['l'], 'rn': ['m'], 'а': ['a']}

def keyboard_adjacent_check(c1, c2):
    return c1.lower() in keyboard_adjacent and c2.lower() in keyboard_adjacent[c1.lower()]

def is_symmetric_mismatch(c1, c2):
    for a, b in symmetric_key_pairs:
        if (c1 == a and c2 == b) or (c1 == b and c2 == a):
            return True
    return False

def is_visual_homoglyph(c1, c2):
    for a, b in homoglyph_pairs:
        if (c1 == a and c2 == b) or (c1 == b and c2 == a):
            return True
    return False

def is_valid_tld(tld):
    return re.fullmatch(r'[a-z]+', tld) is not None  # 正規表現でTLDっぽさを判定（英字のみ）

def classify_edit_ops_japanese(correct, typo):
    # 転置（入力順序ミス）の単独判定（優先的に）
    if damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1:
        return "入力順序ミス"

    ops = Levenshtein.editops(correct, typo)
    if not ops:
        return "一致"

    causes = set()

    for op in ops:
        tag, src_i, tgt_i = op
        c1 = correct[src_i] if src_i < len(correct) else ''
        c2 = typo[tgt_i] if tgt_i < len(typo) else ''

        if tag == 'replace':
            if keyboard_adjacent_check(c1, c2):
                causes.add("隣接キー誤打")
            elif is_symmetric_mismatch(c1, c2):
                causes.add("左右対称キー誤打")
            elif is_visual_homoglyph(c1, c2):
                causes.add("ホモグリフ（視覚類似文字）")
            else:
                causes.add("スペルミス（認知ミス）")

        elif tag == 'insert':
            # .co → .co.jp のようなTLD追加
            if correct.endswith('.co') and typo.endswith('.co.jp'):
                causes.add("別TLD結合")
            # .co.jp → cojp のようなドット忘れ（別LD結合）
            elif correct.endswith('.co.jp') and typo.endswith('cojp'):
                causes.add("別LD結合（ドット忘れ）")
            else:
                causes.add("二重入力")

        elif tag == 'delete':
            causes.add("入力漏れ")

    # 順序入れ替えかどうかの追加判定
    if damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1:
        causes.add("入力順序ミス")

    # ドット抜け（別LD結合）判定
    if '.' in correct and '.' not in typo:
        causes.add("別LD結合・ドット抜け")

    # TLD違いの詳細判定（例: .jp → .jo）
    correct_tld = correct.split('.')[-1]
    typo_tld = typo.split('.')[-1] if '.' in typo else ''
    if typo_tld and typo_tld != correct_tld and is_valid_tld(typo_tld):
        if keyboard_adjacent_check(correct_tld, typo_tld):
            causes.add("隣接キー誤打")
        else:
            causes.add("別TLD結合")

    return '・'.join(sorted(causes))



# --------------原因別集計-----------------
def classify_edit_ops_japanese(correct, typo):
    # 転置（入力順序ミス）の単独判定（優先的に）
    if damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1:
        return "入力順序ミス"

    ops = Levenshtein.editops(correct, typo)
    if not ops:
        return "一致"

    causes = set()

    for op in ops:
        tag, src_i, tgt_i = op
        c1 = correct[src_i] if src_i < len(correct) else ''
        c2 = typo[tgt_i] if tgt_i < len(typo) else ''

        if tag == 'replace':
            if keyboard_adjacent_check(c1, c2):
                causes.add("隣接キー誤打")
            elif is_symmetric_mismatch(c1, c2):
                causes.add("左右対称キー誤打")
            elif is_visual_homoglyph(c1, c2):
                causes.add("ホモグリフ・視覚類似文字")
            else:
                causes.add("スペルミス（認知ミス）")

        elif tag == 'insert':
            # .co → .co.jp のようなTLD追加
            if correct.endswith('.co') and typo.endswith('.co.jp'):
                causes.add("別TLD結合")
            # .co.jp → cojp のようなドット忘れ（別LD結合）
            elif correct.endswith('.co.jp') and typo.endswith('cojp'):
                causes.add("別LD結合（ドット忘れ）")
            else:
                causes.add("二重入力")

        elif tag == 'delete':
            causes.add("入力漏れ")

    # 順序入れ替えかどうかの追加判定
    if damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1:
        causes.add("入力順序ミス")

    # ドット抜け（別LD結合）判定
    if '.' in correct and '.' not in typo:
        causes.add("別LD結合・ドット抜け")

    # TLD違いの詳細判定（例: .jp → .jo）
    correct_tld = correct.split('.')[-1]
    typo_tld = typo.split('.')[-1] if '.' in typo else ''
    if typo_tld and typo_tld != correct_tld and is_valid_tld(typo_tld):
        if keyboard_adjacent_check(correct_tld, typo_tld):
            causes.add("隣接キー誤打")
        else:
            causes.add("別TLD結合")

    return '・'.join(sorted(causes))

# メイン処理
def append_typo_causes(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    causes = []
    for _, row in df.iterrows():
        correct = extract_domain(str(row['correct_address']))
        typo = extract_domain(str(row['input_address']))
        cause = classify_edit_ops_japanese(correct, typo)
        causes.append(cause)

    df['cause'] = causes

    # CSV に出力
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig") #UTF-8エンコードで出力し、Excelで開く時に明示的に「UTF-8」を指定
    # print(f"→ '{output_csv_path}' に出力しました。") ##→ 'domaintypos_dl4_causes.csv' に出力しました。

# 実行部分（必要に応じてファイル名を変更）
append_typo_causes("filtered_domain_typos_dl4.csv", "domaintypos_dl4_causes.csv")

def keyboard_adjacent_check(c1, c2):
    return c1.lower() in keyboard_adjacent and c2.lower() in keyboard_adjacent[c1.lower()]

def is_visual_homoglyph(c1, c2):
    return any((c1 == a and c2 == b) or (c1 == b and c2 == a) for a, b in homoglyph_pairs)

def is_symmetric_mismatch(c1, c2):
    return any((c1 == a and c2 == b) or (c1 == b and c2 == a) for a, b in symmetric_key_pairs)

def is_valid_tld(tld):
    return len(tld) in [2, 3, 4] and tld.isalpha()

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
                causes.add("ホモグリフ・視覚類似文字")
            else:
                causes.add("スペルミス（認知ミス）")

        elif tag == 'insert':
            correct_parts.append('')
            typo_parts.append(c2)
            if correct.endswith('.co') and typo.endswith('.co.jp'):
                causes.add("別TLD結合")
            elif correct.endswith('.co.jp') and typo.endswith('cojp'):
                causes.add("別LD結合（ドット忘れ）")
            else:
                causes.add("二重入力")

        elif tag == 'delete':
            correct_parts.append(c1)
            typo_parts.append('')
            causes.add("入力漏れ")

    # ドット忘れ
    if '.' in correct and '.' not in typo:
        causes.add("別LD結合・ドット抜け")

    # TLD違い
    correct_tld = correct.split('.')[-1]
    typo_tld = typo.split('.')[-1] if '.' in typo else ''
    if typo_tld and typo_tld != correct_tld and is_valid_tld(typo_tld):
        if keyboard_adjacent_check(correct_tld, typo_tld):
            causes.add("隣接キー誤打")
        else:
            causes.add("別TLD結合")

    return {
        "cause": '・'.join(sorted(causes)),
        "correct_part": ' '.join(correct_parts),
        "mismatched_part": ' '.join(typo_parts)
    }

# メイン処理
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

# 実行部分
append_typo_causes("filtered_domain_typos_dl4.csv", "domaintypos_dl4_causes2.csv")

# 差分の抽出（difflib を使用して 2文字以上も対応）
def extract_ngram_diffs(correct, typo):
    sm = difflib.SequenceMatcher(None, correct, typo)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        src = correct[i1:i2]  # 元
        tgt = typo[j1:j2]     # 誤り
        diffs.append((tag, src, tgt))
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

        diffs = extract_ngram_diffs(correct, typo)

        for cause in causes:
            for tag, c1, c2 in diffs:
                # 空白対策
                c1 = c1 or '（空）'
                c2 = c2 or '（空）'
                cause_diff_counter[cause][(c1, c2)] += 1

    # 出力
    for cause, counter in cause_diff_counter.items():
        print(f"\n【原因: {cause}】")
        for (c1, c2), count in counter.most_common():
            print(f"  {c1} → {c2:<10}: {count}件")

# 実行
analyze_ngram_differences("domaintypos_dl4_causes2.csv")


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

#==========================引用終わり==================================

# ----------タイポ候補生成器---------------
typo_weights = cause_ratios  ##集計結果の割合 (cause_ratios) を typo_weights に割り当てる

def typo_generator(domain, top_n=30):
    variants = []

    for i in range(len(domain)):
        c = domain[i]

        # 入力漏れ
        variants.append((domain[:i] + domain[i+1:], ["入力漏れ"]))

        # 二重入力
        variants.append((domain[:i] + c + c + domain[i+1:], ["二重入力"]))

        # 隣接キー誤打
        for adj in keyboard_adjacent.get(c.lower(), ''):
            variants.append((domain[:i] + adj + domain[i+1:], ["隣接キー誤打"]))

        # ホモグリフ・視覚類似文字
        for base, glyphs in homoglyphs.items():
            if c == base:
                for g in glyphs:
                    variants.append((domain[:i] + g + domain[i+1:], ["ホモグリフ・視覚類似文字"]))
            elif c in glyphs:
                variants.append((domain[:i] + base + domain[i+1:], ["ホモグリフ・視覚類似文字"]))

        # 左右対称キー誤打
        for a, b in symmetric_key_pairs:
            if c == a:
                variants.append((domain[:i] + b + domain[i+1:], ["左右対称キー誤打"]))
            elif c == b:
                variants.append((domain[:i] + a + domain[i+1:], ["左右対称キー誤打"]))

    # 入力順序ミス
    for i in range(len(domain) - 1):
        swapped = domain[:i] + domain[i+1] + domain[i] + domain[i+2:]
        variants.append((swapped, ["入力順序ミス"]))

    # 別TLD結合
    if domain.endswith(".co.jp"):
        variants.append((domain.replace(".co.jp", ".com"), ["別TLD結合"]))
    elif domain.endswith(".com"):
        variants.append((domain.replace(".com", ".co.jp"), ["別TLD結合"]))

    # ドット抜け／別LD結合
    if '.' in domain:
        dotless = domain.replace('.', '')
        variants.append((dotless, ["ドット抜け", "別LD結合（ドット忘れ）"]))

    # 統合とスコア集計
    seen = {}
    for typo, causes in variants:
        score = sum(typo_weights.get(c, 0) for c in causes)
        if typo not in seen or score > seen[typo][1]:
            seen[typo] = (causes, score)

    ranked = sorted(seen.items(), key=lambda x: x[1][1], reverse=True)
    return [{
        "typo": typo,
        "causes": causes,
        "score": round(score, 3)
    } for typo, (causes, score) in ranked[:top_n]]


# 実行部分を修正
results = typo_generator("treasurefactory.co.jp")

# ヘッダーを出力
print("==============================================================================")
print("■ タイポドメイン生成【タイポ候補,  原因,  スコア】")

# 結果をカンマ区切り形式で出力
for r in results:
    # 原因リストを '・' 区切りの文字列に変換
    cause_str = '・'.join(r['causes'])
    
    # データをカンマ区切りで整形
    output_line = f"'{r['typo']}',  '{cause_str}',  {r['score']}"
    print(output_line)

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

# ==============================================================================
# 🗄️ 1. 定数と汎用ヘルパー関数
# ==============================================================================

keyboard_adjacent = { # キーボード隣接キーマップ (アドレス全体で使用)
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qws', 's': 'qwedazx', 'd': 'erfcsx', 'f': 'rtdgvcj',
    'g': 'tyfhvbn', 'h': 'yugjnb', 'j': 'uikhmnf', 'k': 'ijolm', 'l': 'okp',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk,',
    '.': ',/', ',': 'm.', '-': '^'
}
symmetric_key_pairs = [('f', 'j'), ('d', 'k'), ('s', 'l'), ('a', ';')] 
homoglyph_pairs = [('1', 'l'), ('0', 'o'), ('i', 'l'), ('rn', 'm'), ('а', 'a'), ('b', 'd'), ('o', 'a')] 
HOMOGLYPHS_FOR_GENERATOR = {'1': ['l'], '0': ['o'], 'i': ['l'], 'l': ['i'], 'r': ['m'], 'b': ['d'], 'd': ['b']} 

def extract_domain(email):
    return email.split('@')[1] if '@' in email else ''

def remove_at(address: str) -> str:
    """アドレスから@記号を削除する（分析用）。"""
    return address.replace('@', '')

def get_mismatched_part(correct, input_):
    matcher = SequenceMatcher(None, correct, input_)
    mismatched = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            mismatched.append(input_[j1:j2])
    return ''.join(mismatched)

def keyboard_adjacent_check(c1, c2):
    return c1.lower() in keyboard_adjacent and c2.lower() in keyboard_adjacent[c1.lower()]

def is_symmetric_mismatch(c1, c2): 
    return any((c1 == a and c2 == b) or (c1 == b and c2 == a) for a, b in symmetric_key_pairs)

def is_visual_homoglyph(c1, c2): 
    return any((c1 == a and c2 == b) or (c1 == b and c2 == a) for a, b in homoglyph_pairs)

def is_valid_tld(tld):
    return len(tld) in [2, 3, 4] and tld.isalpha()

def get_transposed_pair(correct, typo):
    """入力順序ミスが単独で発生している場合の文字ペアを返す"""
    if damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1:
        for i in range(len(correct) - 1):
            if correct[i] == typo[i+1] and correct[i+1] == typo[i] and correct[:i] == typo[:i] and correct[i+2:] == typo[i+2:]:
                return (correct[i], correct[i+1])
    return None

def identify_single_replacement(correct, typo):
    """DL距離1の置換・挿入・削除ミスの差分文字ペアを特定する（予測生成用）。"""
    matcher = SequenceMatcher(None, correct, typo)
    ops = matcher.get_opcodes()
    
    replacements = [(correct[i1:i2], typo[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag == 'replace']
    inserts = [(typo[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag == 'insert']
    deletes = [(correct[i1:i2]) for tag, i1, i2, j1, j2 in ops if tag == 'delete']
    
    if len(replacements) == 1 and not inserts and not deletes:
        c1 = replacements[0][0]; c2 = replacements[0][1]
        if len(c1) == 1 and len(c2) == 1: return (c1, c2)  # 置換
            
    if len(inserts) == 1 and not replacements and not deletes:
        return ('（空）', inserts[0][0]) # 挿入 (二重入力)

    if len(deletes) == 1 and not replacements and not inserts:
        return (deletes[0][0], '（空）') # 削除 (入力漏れ/ドット抜け)

    return ('', '') # 複合または非距離1ミス


# ==============================================================================
# 🎯 2. フィルタリング、分類、付与、差分抽出の関数
# ==============================================================================

def filter_address_differences(input_path, output_path, threshold=5):
    """アドレス全体を対象にタイポデータを抽出し、整形する。（@を削除してDL距離を計算）"""
    df = pd.read_csv(input_path)

    # ⬇️ 修正: @を削除した文字列を比較対象にする (ノイズ対策)
    df["correct_target"] = df["correct_address"].astype(str).apply(remove_at)
    df["input_target"] = df["input_address"].astype(str).apply(remove_at)

    # Damerau-Levenshtein距離を計算 (アドレス全体から@を除いた部分で計算)
    df["edit_distance"] = df.apply(
        lambda row: damerau_levenshtein_distance(row["correct_target"], row["input_target"]),
        axis=1
    )

    filtered_df = df[
        (df["edit_distance"] > 0) & (df["edit_distance"] <= threshold)
    ].reset_index(drop=True)

    # 差分部分（mismatched_part）の抽出 (元のaddress列を使用)
    filtered_df["mismatched_part"] = filtered_df.apply(
        lambda row: get_mismatched_part(row["correct_target"], row["input_target"]),
        axis=1
    )

    final_df = filtered_df[["user_id", "step_id", "correct_address", "input_address", "edit_distance", "mismatched_part"]]

    final_df.to_csv(output_path, index=False)
    print(f"[INFO] アドレス全体のタイポを抽出しました: {output_path}")


def classify_edit_ops_japanese(correct, typo):
    """@を削除したアドレス全体を比較し、原因を分類する。"""
    correct_parts = []; typo_parts = []; causes = set()

    # @はすでに削除済みのため、TLD関連の処理は不要

    if damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1:
        causes.add("入力順序ミス")

    ops = Levenshtein.editops(correct, typo)
    for tag, src_i, tgt_i in ops:
        c1 = correct[src_i] if src_i < len(correct) else ''
        c2 = typo[tgt_i] if tgt_i < len(typo) else ''

        if tag == 'replace':
            correct_parts.append(c1); typo_parts.append(c2)
            if keyboard_adjacent_check(c1, c2): causes.add("隣接キー誤打")
            elif is_symmetric_mismatch(c1, c2): causes.add("左右対称キー誤打")
            elif is_visual_homoglyph(c1, c2): causes.add("ホモグリフ（視覚類似文字）")
            else: causes.add("スペルミス（認知ミス）")

        elif tag == 'insert':
            correct_parts.append(''); typo_parts.append(c2)
            causes.add("二重入力")

        elif tag == 'delete':
            correct_parts.append(c1); typo_parts.append('')
            if c1 == '.': causes.add("ドット抜け")
            else: causes.add("入力漏れ")

    return {
        "cause": '・'.join(sorted(causes)),
        "correct_part": ' '.join(correct_parts),
        "mismatched_part": ' '.join(typo_parts)
    }


def append_typo_causes_full_address(input_csv_path, output_csv_path):
    """アドレス全体 (ただし@なし) を比較対象として原因を付与し、CSVに出力する。"""
    df = pd.read_csv(input_csv_path)

    causes = []
    correct_parts = []
    typo_parts = []

    for _, row in df.iterrows():
        # @を削除した文字列を分類に渡す
        correct = remove_at(str(row['correct_address']))
        typo = remove_at(str(row['input_address']))
        
        result = classify_edit_ops_japanese(correct, typo)
        causes.append(result["cause"])
        correct_parts.append(result["correct_part"])
        typo_parts.append(result["mismatched_part"])

    df['cause'] = causes
    df['correct_part'] = correct_parts
    df['mismatched_part'] = typo_parts

    desired_columns = [
        'user_id', 'step_id', 'correct_address', 'input_address', 'edit_distance',
        'correct_part', 'mismatched_part', 'cause'
    ]

    df = df[[col for col in desired_columns if col in df.columns]]
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] アドレス全体分析の結果を保存しました: {output_csv_path}")


def extract_ngram_diffs(correct, typo):
    """difflibベースでNグラム差分を抽出し、空文字列を'（空）'に変換して返す。"""
    sm = difflib.SequenceMatcher(None, correct, typo)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal': continue
        src = correct[i1:i2] or '（空）'
        tgt = typo[j1:j2] or '（空）'
        diffs.append((src, tgt)) 
    return diffs


# ==============================================================================
# 📊 4. 統合分析関数 (レポート機能付き)
# ==============================================================================

def analyze_for_ranking(csv_path: str) -> Tuple[Dict[str, float], Dict[str, Dict[Any, float]]]:
    """ランキング用の個別ミス重み (W_individual) を計算し、レポートを出力する。"""
    df = pd.read_csv(csv_path)
    
    individual_rank_weights = defaultdict(dict)
    all_causes = []
    cause_diff_counter = defaultdict(Counter)

    for _, row in df.iterrows():
        # @を削除した文字列で差分を抽出 (NOTE: dfの'correct_address'列には@が含まれる)
        correct_at_free = remove_at(str(row['correct_address']))
        typo_at_free = remove_at(str(row['input_address']))
        
        cause_field = str(row['cause'])
        causes = [c.strip() for c in cause_field.split('・')]

        diffs = extract_ngram_diffs(correct_at_free, typo_at_free)
        all_causes.extend(causes)

        for cause in causes:
            # 入力順序ミスの特別処理
            if cause == "入力順序ミス":
                transposed_pair = get_transposed_pair(correct_at_free, typo_at_free)
                if transposed_pair:
                    c1, c2 = transposed_pair
                    # キーは文字列形式で統一
                    key = f'{c1} {c2} -> {c2} {c1}'
                    cause_diff_counter[cause][key] += 1
            else:
                for c1, c2 in diffs:
                    # その他の原因は、差分パターンをタプルキーで集計
                    cause_diff_counter[cause][(c1, c2)] += 1

    # --- 1. 割合と内部重みの計算 ---
    cause_counts = Counter(all_causes)
    total_major_events = sum(cause_counts.values())
    major_ratios = {k: round(v / total_major_events, 3) for k, v in cause_counts.items()}

    total_typo_events = sum(sum(counter.values()) for counter in cause_diff_counter.values())
    
    for cause, counter in cause_diff_counter.items():
        for key, count in counter.items():
            rank_score = count / total_typo_events
            individual_rank_weights[cause][key] = rank_score
            
    # --- 2. レポート出力 ---
    print("\n" + "=" * 78)
    print("■ 差分パターン上位レポート:\n")
    for cause, counter in cause_diff_counter.items():
        print(f"\n【原因: {cause}】")
        for key, count in counter.most_common(5): 
            # キーの形式を判別して出力
            if isinstance(key, str): 
                 print(f"  {key:<10}: {count}件")
            else: 
                 print(f"  {key[0]} → {key[1]:<10}: {count}件")

    print("\n" + "=" * 78)
    print("■ 原因別集計と割合（重み）:\n")
    for cause in sorted(cause_counts, key=cause_counts.get, reverse=True):
        count = cause_counts[cause]
        ratio = major_ratios[cause]
        print(f"{cause:<25}: {count:>5}件 ({ratio:>5.3f})")
    print("=" * 78)
    
    return major_ratios, individual_rank_weights


# ==============================================================================
# 🚀 5. メイン実行ブロック (アドレス全体モード用)
# ==============================================================================

if __name__ == "__main__":
    
    INPUT_FILE = "filtered_address.csv" # 元の入力ファイル
    ADDRESS_MODE_FILE = "filtered_address_full.csv" # 新規中間ファイル
    CAUSES_CSV_FILE = "fulladdress_causes.csv" # 最終分析ファイル

    print("--- アドレス全体分析パイプライン開始 ---")
    
    # 1. データ抽出とフィルタリング (アドレス全体モード)
    filter_address_differences(INPUT_FILE, ADDRESS_MODE_FILE, threshold=4)

    # 2. 原因分類を付与しCSVに出力 (アドレス全体モード)
    append_typo_causes_full_address(ADDRESS_MODE_FILE, CAUSES_CSV_FILE) 
    
    # 3. 分析の実行: 個別ミスの重み計算とレポート出力
    major_weights, individual_rank_weights = analyze_for_ranking(CAUSES_CSV_FILE)
    
    # 4. ドメインランキング生成の実行 (この部分の関数定義は省略)
    if major_weights and individual_rank_weights:
        # 例として、重み計算が成功したことを表示
        print(f"\n[SUCCESS] 個別重み（W_individual）の計算とレポート出力が完了しました。")
        print(f"総イベント数に基づくトップの個別ミススコア: {max(max(d.values()) for d in individual_rank_weights.values()):.6f}")
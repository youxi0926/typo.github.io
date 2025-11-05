import os
import re
import Levenshtein
from pyxdameraulevenshtein import damerau_levenshtein_distance
import pandas as pd
from difflib import SequenceMatcher
import difflib
from collections import defaultdict, Counter
import json
from typing import Dict, Tuple, Any, List, Set # 型ヒントを使用

# ==============================================================================
# 🗄️ 1. 定数と汎用ヘルパー関数
# ==============================================================================

# キーボード隣接キーマップ (QWERTY)
keyboard_adjacent = { 
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
    """メールアドレスからドメイン部を抽出。"""
    return email.split('@')[1] if '@' in email else ''

def get_mismatched_part(correct, input_):
    """difflibを使用し、入力側（input_）の誤入力部分を抽出。"""
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
    """入力順序ミスが単独で発生している場合の文字ペアを返す (DL距離=1, L距離>1)。"""
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
        if len(c1) == 1 and len(c2) == 1: return (c1, c2)
            
    if len(inserts) == 1 and not replacements and not deletes:
        return ('（空）', inserts[0][0]) 

    if len(deletes) == 1 and not replacements and not inserts:
        return (deletes[0][0], '（空）')

    return ('', '') 

# ==============================================================================
# 🎯 2. コア機能: 分類とデータ準備
# ==============================================================================

def filter_domain_differences_with_mismatch(input_path, output_path, threshold=5):
    """タイポデータの抽出、フィルタリング、初期整形を行う。"""
    df = pd.read_csv(input_path)
    df["correct_domain"] = df["correct_address"].astype(str).apply(extract_domain)
    df["input_domain"] = df["input_address"].astype(str).apply(extract_domain)

    df["domain_edit_distance"] = df.apply(
        lambda row: damerau_levenshtein_distance(row["correct_domain"], row["input_domain"]), axis=1
    )

    filtered_df = df[
        (df["domain_edit_distance"] > 0) & (df["domain_edit_distance"] <= threshold)
    ].reset_index(drop=True)

    filtered_df["mismatched_part"] = filtered_df.apply(
        lambda row: get_mismatched_part(row["correct_domain"], row["input_domain"]), axis=1
    )

    filtered_df["edit_distance"] = filtered_df["domain_edit_distance"]
    final_df = filtered_df[["user_id", "step_id", "correct_address", "input_address", "edit_distance", "mismatched_part"]]
    final_df.to_csv(output_path, index=False)
    print(f"[INFO] 1. ドメインの違いと差分を出力しました: {output_path}")


def classify_edit_ops_japanese(correct, typo):
    """ドメイン部を比較し、原因を分類する。"""
    correct_parts = []; typo_parts = []; causes = set()

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

    correct_tld = correct.split('.')[-1]
    typo_tld = typo.split('.')[-1] if '.' in typo else ''
    if typo_tld and typo_tld != correct_tld and is_valid_tld(typo_tld):
        if keyboard_adjacent_check(correct_tld, typo_tld): causes.add("隣接キー誤打")

    return {
        "cause": '・'.join(sorted(causes)),
        "correct_part": ' '.join(correct_parts),
        "mismatched_part": ' '.join(typo_parts)
    }


def append_typo_causes(input_csv_path, output_csv_path):
    """原因、Correct Part, Mismatched Part を付与しCSV出力する。"""
    df = pd.read_csv(input_csv_path)

    results = df.apply(
        lambda row: classify_edit_ops_japanese(
            extract_domain(str(row['correct_address'])),
            extract_domain(str(row['input_address']))
        ), axis=1, result_type='expand'
    )
    
    df = pd.concat([df, results], axis=1)

    desired_columns = [
        'user_id', 'step_id', 'correct_address', 'input_address', 'edit_distance',
        'correct_part', 'mismatched_part', 'cause'
    ]
    df = df[[col for col in desired_columns if col in df.columns]]
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 2. 原因分類を付与しました: {output_csv_path}")


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
# 📊 3. 統合分析関数 (レポート機能と個別重み計算)
# ==============================================================================

def analyze_for_ranking(csv_path: str) -> Tuple[Dict[str, float], Dict[str, Dict[Tuple[Any, Any], float]]]:
    """ランキング用の個別ミス重み (W_individual) を計算し、レポートを出力する。"""
    df = pd.read_csv(csv_path)
    
    individual_rank_weights = defaultdict(dict)
    all_causes = []
    cause_diff_counter = defaultdict(Counter)

    for _, row in df.iterrows():
        correct = extract_domain(str(row['correct_address']))
        typo = extract_domain(str(row['input_address']))
        cause_field = str(row['cause'])
        causes = [c.strip() for c in cause_field.split('・')]

        # 入力順序ミスの特別処理 (ここで文字列キーを cause_diff_counter に直接格納)
        if "入力順序ミス" in causes:
            transposed_pair = get_transposed_pair(correct, typo)
            if transposed_pair:
                c1, c2 = transposed_pair
                key = f'{c1} {c2} -> {c2} {c1}'
                cause_diff_counter["入力順序ミス"][key] += 1
                # 複合ミスでない場合、他の diffs の処理をスキップして重複を回避
                if len(causes) == 1: continue 

        diffs = extract_ngram_diffs(correct, typo)
        all_causes.extend(causes)

        for cause in causes:
            if cause == "入力順序ミス": continue # 特殊処理で対応済みのためスキップ
            
            for c1, c2 in diffs:
                # 差分パターンをそのまま集計 (タプルキー)
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
            
    # --- 2. レポート出力 (ご要望の形式に統合) ---
    print("\n" + "=" * 78)
    print("■ 3. 原因別集計と割合（重み）:\n")
    for cause in sorted(cause_counts, key=cause_counts.get, reverse=True):
        count = cause_counts[cause]
        ratio = major_ratios[cause]
        print(f"{cause:<25}: {count:>5}件 ({ratio:>5.3f})")
    
    print("\n" + "=" * 78)
    print("■ 4. 差分パターン上位レポート:\n")
    for cause, counter in cause_diff_counter.items():
        print(f"\n【原因: {cause}】")
        for key, count in counter.most_common(5): 
            if isinstance(key, str): 
                 print(f"  {key:<10}: {count}件")
            else: 
                 print(f"  {key[0]} → {key[1]:<10}: {count}件")
    print("=" * 78)
    
    return major_ratios, individual_rank_weights


# --------------------------------------------------------------------------
# ⭐ NEW: タプルキーをJSONフレンドリーな文字列に変換するヘルパー関数
# --------------------------------------------------------------------------
def convert_internal_keys_to_str(individual_weights: Dict[str, Dict[Tuple[Any, Any], float]]) -> Dict[str, Dict[str, float]]:
    """individual_rank_weights内のタプルキーをJSONフレンドリーな文字列キーに変換する。"""
    converted_weights = {}
    for cause, inner_dict in individual_weights.items():
        converted_inner_dict = {}
        for key, score in inner_dict.items():
            # キーがタプルであれば join('') で文字列に変換、文字列であればそのまま使用
            if isinstance(key, tuple):
                # JSONでは '（空）' も文字列として扱う必要があるため、そのまま結合
                key_str = "".join(key) 
            else:
                key_str = key 
            
            converted_inner_dict[key_str] = score
        converted_weights[cause] = converted_inner_dict
    return converted_weights


# ==============================================================================
# 🎯 5. 予測モデル (スコアリング関数)
# ==============================================================================

def typo_generator_ranked(domain: str, individual_weights: dict, top_n: int = 30):
    """
    個別ミス件数/全タイポ件数 (W_individual) でスコアリングし、ランキングする。
    """
    variants = defaultdict(lambda: (set(), 0)) 

    # --- A. タイポ候補の生成 ---
    for i in range(len(domain)):
        c = domain[i]
        cLower = c.lower()
        
        # 1. 入力漏れ (Deletion)
        delTy = domain[:i] + domain[i+1:]
        # ⬇️ 修正箇所: 三項演算子をPython形式に修正 ⬇️
        causes_for_delete = "ドット抜け" if c == '.' else "入力漏れ"

        # 2. 二重入力 (Insertion/Repetition)
        dupTy = domain[:i] + c + c + domain[i+1:]
        variants[dupTy][0].add("二重入力")

        # 3. 隣接キー誤打 (Substitution)
        if cLower in keyboard_adjacent:
            for adj in keyboard_adjacent[cLower]:
                adjTy = domain[:i] + adj + domain[i+1:]
                variants[adjTy][0].add("隣接キー誤打")

        # 4. ホモグリフ・視覚類似文字 (Substitution)
        if cLower in HOMOGLYPHS_FOR_GENERATOR:
            for g in HOMOGLYPHS_FOR_GENERATOR[cLower]:
                 hgTy = domain[:i] + g + domain[i+1:]
                 variants[hgTy][0].add("ホモグリフ（視覚類似文字）")
            
        # 5. 左右対称キー誤打 (Substitution)
        for a, b in symmetric_key_pairs:
            if c == a: variants[domain[:i] + b + domain[i+1:]][0].add("左右対称キー誤打")
            elif c == b: variants[domain[:i] + a + domain[i+1:]][0].add("左右対称キー誤打")

        # 6. 入力順序ミス (Transposition)
        if i < len(domain) - 1:
            swapped = domain[:i] + domain[i+1] + domain[i] + domain[i+2:]
            variants[swapped][0].add("入力順序ミス")

        # 7. ドット抜け (Deletion of dot - TLDのドットは分類で処理)
        if c == '.':
            delDot = domain[:i] + domain[i+1:]
            variants[delDot][0].add("ドット抜け")

    # --- B. スコアリングとランキング ---
    ranked_results = []
    
    for typo, (causes, _) in variants.items():
        if typo == domain: continue
        
        final_score = 0
        c1, c2 = identify_single_replacement(domain, typo) 
        
        for cause in causes:
            W_individual = 0.0
            
            # 内部重み (W_individual) の参照ロジック
            # キーの形式は 'po' や 'a（空）'
            key = None
            if cause == "入力順序ミス":
                transposed_pair = get_transposed_pair(domain, typo)
                if transposed_pair:
                    k1, k2 = transposed_pair
                    key = f'{k1} {k2} -> {k2} {k1}' # 文字列キー
            elif c1 and c2: # 置換、挿入、削除の単一ミス
                key = c1 + c2 if c1 != '（空）' else '（空）' + c2
            
            if key and cause in individual_weights:
                # 逆順チェック (置換系)
                W_individual = individual_weights[cause].get(key, 0.0)
                if W_individual == 0.0 and len(key) == 2 and '（空）' not in key:
                    W_individual = individual_weights[cause].get(key[::-1], 0.0)
            
            final_score += W_individual

        # 複合ミス ペナルティ
        if len(causes) > 1:
            final_score *= 0.5 

        distance = damerau_levenshtein_distance(domain, typo)

        ranked_results.append({
            "typo": typo,
            "causes": '・'.join(sorted(causes)),
            "score": round(final_score, 6), # 精密なスコアを出力
            "distance": distance
        })

    ranked_results.sort(key=lambda x: (x['score'], -x['distance']), reverse=True)

    return ranked_results[:top_n]


# ==============================================================================
# 🚀 6. メイン実行ブロック
# ==============================================================================

if __name__ == "__main__":
    
    # --------------------------------------------------------------------------
    # 必須ファイルパス設定
    INPUT_FILE = "filtered_address.csv"
    DL4_FILTERED_FILE = "filtered_domain_typos_dl4.csv"
    CAUSES_CSV_FILE = "domaintypos_dl4_causes2.csv"
    
    # 1. データ抽出とフィルタリング (ドメイン部のみ)
    filter_domain_differences_with_mismatch(INPUT_FILE, DL4_FILTERED_FILE, threshold=4)

    # 2. 原因分類を付与しCSVに出力
    append_typo_causes(DL4_FILTERED_FILE, CAUSES_CSV_FILE)
    
    # 3. 分析の実行: 個別ミスの重み (W_individual) を計算
    major_weights, individual_rank_weights = analyze_for_ranking(CAUSES_CSV_FILE)
    
    # 4. Web用データのエクスポート (JSON変換を適用)
    converted_individual_weights = convert_internal_keys_to_str(individual_rank_weights)

    OUTPUT_JSON_FILE = "data.json"
    web_data_export = {
        "major_weights": major_weights, 
        "individual_weights": converted_individual_weights, 
        "keyboard_adjacent": keyboard_adjacent,
        "symmetric_key_pairs": [list(pair) for pair in symmetric_key_pairs],
        "homoglyph_pairs": [list(pair) for pair in homoglyph_pairs],
        "homoglyphs_for_generator": HOMOGLYPHS_FOR_GENERATOR
    }
    
    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(web_data_export, f, indent=4, ensure_ascii=False)
        print(f"\n[INFO] Web用データのエクスポート完了: {OUTPUT_JSON_FILE}")
    except Exception as e:
        print(f"\n[ERROR] JSONエクスポート中にエラーが発生しました: {e}")

    # 5. コンソールでのランキング生成 (最終テスト)
    correct_domain = input("\n入力されたドメインのタイポドメイン候補を生成する（例: treasurefactory.co.jp）: ").strip()
    
    if major_weights and individual_rank_weights:
        predicted_typos = typo_generator_ranked(
            domain=correct_domain,
            individual_weights=individual_rank_weights,
            top_n=10
        )
        
        print("\n" + "=" * 78)
        print(f"🥇 '{correct_domain}' に対する予測タイポドメインランキング:\n")
        for i, r in enumerate(predicted_typos):
            print(f"{i+1}位 {r['typo']:<30} (スコア: {r['score']:.6f}, 距離: {r['distance']}, 原因: {r['causes']})")
    else:
        print("\n[エラー] 重みデータが計算されていないため、ランキング生成を実行できませんでした。")
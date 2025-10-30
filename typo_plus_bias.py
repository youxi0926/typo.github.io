import os
import re
import Levenshtein
from pyxdameraulevenshtein import damerau_levenshtein_distance
import pandas as pd
from difflib import SequenceMatcher
import difflib
from collections import defaultdict, Counter

# ==============================================================================
# 🗄️ 1. 定数とヘルパー関数
# ==============================================================================

# NEW: 頻出隣接キー置換ペア (観測データに基づき調整)
FREQUENT_ADJACENT_PAIRS = {('.', ','), ('p', 'o'), ('o', 'p'), ('n', 'm'), ('m', 'n'), ('r', 't'), ('t', 'r'), (',', '.'), ('o', 'p'), ('p', 'o')} 
HOMOGLYPHS_FOR_GENERATOR = {'1': ['l'], '0': ['o'], 'i': ['l'], 'l': ['i'], 'r': ['m'], 'b': ['d'], 'd': ['b']} 
TLD_ALTERNATIVES = {'com': ['co.jp', 'net'], 'co.jp': ['com', 'co'], 'net': ['com', 'co.jp']}

keyboard_adjacent = { 
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qws', 's': 'qwedazx', 'd': 'erfcsx', 'f': 'rtdgvcj',
    'g': 'tyfhvbn', 'h': 'yugjnb', 'j': 'uikhmnf', 'k': 'ijolm', 'l': 'okp',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk,',
    '.': ',/', ',': 'm.',
    '-': '^'
}

symmetric_key_pairs = [('f', 'j'), ('d', 'k'), ('s', 'l'), ('a', ';')] 
homoglyph_pairs = [('1', 'l'), ('0', 'o'), ('i', 'l'), ('rn', 'm'), ('а', 'a'), ('b', 'd')] 

def extract_domain(email):
    return email.split('@')[1] if '@' in email else ''

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

# ⭐ 修正不要: DL距離1の置換ペアを特定する関数
def identify_single_replacement(correct, typo):
    """DL距離1の置換ミスの文字ペアを特定する。"""
    matcher = SequenceMatcher(None, correct, typo)
    ops = matcher.get_opcodes()
    
    # 置換が一つだけ発生していることを確認
    replacements = [(correct[i1:i2], typo[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag == 'replace']
    inserts = [tag for tag, i1, i2, j1, j2 in ops if tag == 'insert']
    deletes = [tag for tag, i1, i2, j1, j2 in ops if tag == 'delete']
    
    if len(replacements) == 1 and not inserts and not deletes:
        c1 = replacements[0][0]
        c2 = replacements[0][1]
        if len(c1) == 1 and len(c2) == 1:
            return (c1, c2)
            
    # 挿入、削除、転置ミスの場合は、差分部分を特定
    if len(correct) < len(typo) and len(typo) - len(correct) == 1 and not replacements and not deletes:
        # 挿入 (二重入力)
        for op in ops:
            if op[0] == 'insert':
                # 正しい文字 = （空）, 入力ミス文字 = c2
                return ('（空）', typo[op[2]])

    if len(correct) > len(typo) and len(correct) - len(typo) == 1 and not replacements and not inserts:
        # 削除 (入力漏れ)
        for op in ops:
            if op[0] == 'delete':
                # 正しい文字 = c1, 入力ミス文字 = （空）
                return (correct[op[1]], '（空）')

    return ('', '') # 複数ミスの場合は空を返す


# ==============================================================================
# 🔬 2. タイポ原因の分類ロジック (コア機能)
# ==============================================================================

def classify_edit_ops_japanese(correct, typo):
    """ドメイン間の編集操作を分析し、タイポ原因と差分パーツを辞書で返す。"""
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

# ==============================================================================
# 🛠️ 3. CSV処理関数 (変更なし)
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


# ==============================================================================
# 📊 4. 分析・レポート関数 (変更なし)
# ==============================================================================

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


def analyze_and_report_all(csv_path):
    """原因の割合、個別ミスの内部重みを計算し、レポートを出力する。"""
    df = pd.read_csv(csv_path)
    
    all_causes = []
    cause_diff_counter = defaultdict(Counter)
    
    # ⬇️ ステップ1. 個別ミス重み計算のための集計 ⬇️
    all_raw_typos_by_cause = defaultdict(Counter) # {原因: {('c1', 'c2'): count}}

    for _, row in df.iterrows():
        correct = extract_domain(str(row['correct_address']))
        typo = extract_domain(str(row['input_address']))
        cause_field = str(row['cause'])
        causes = [c.strip() for c in cause_field.split('・')]

        diffs = extract_ngram_diffs(correct, typo)
        
        all_causes.extend(causes)

        for cause in causes:
            # 内部重み計算と差分レポートのため、diffsの全パターンを記録
            for c1, c2 in diffs:
                cause_diff_counter[cause][(c1, c2)] += 1 
                all_raw_typos_by_cause[cause][(c1, c2)] += 1

    # ⬇️ ステップ2. 割合と内部重みの計算 ⬇️
    cause_counts = Counter(all_causes)
    total = sum(cause_counts.values())
    cause_ratios = {k: round(v / total, 3) for k, v in cause_counts.items()} # W_major

    individual_typo_weights = defaultdict(dict) # W_internal
    for cause, counter in all_raw_typos_by_cause.items():
        total_for_cause = sum(counter.values())
        if total_for_cause == 0: continue
            
        for (c1, c2), count in counter.most_common():
            # W_internal = 個別ミスの件数 / その大カテゴリの総件数
            individual_typo_weights[cause][(c1, c2)] = count / total_for_cause

    # --- レポート出力 ---
    print("\n" + "=" * 78)
    print("■ 3. 原因別集計と割合（重み）レポート:\n")
    for cause in sorted(cause_counts, key=cause_counts.get, reverse=True):
        count = cause_counts[cause]
        ratio = cause_ratios[cause]
        print(f"{cause:<25}: {count:>5}件 ({ratio:>5.3f})")
    
    print("\n" + "=" * 78)
    print("■ 4. 原因別 差分パターン上位レポート:\n")
    for cause, counter in cause_diff_counter.items():
        print(f"\n【原因: {cause}】")
        for (c1, c2), count in counter.most_common(5): 
            print(f"  {c1} → {c2:<10}: {count}件")
            
    return cause_ratios, individual_typo_weights


# ==============================================================================
# 🎯 5. 予測モデル (スコアリング関数)
# ==============================================================================

def typo_generator_ranked(domain: str, major_weights: dict, internal_weights: dict, top_n: int = 30):
    """
    高度な重み付けに基づき、発生可能性の高いタイポドメインを生成・ランキングする。
    """
    variants = defaultdict(lambda: (set(), 0)) # {typo_domain: (causes_set, score)}
    
    # ------------------------------------------------------------------
    # A. タイポ候補の生成 (距離1の操作を中心に)
    # ------------------------------------------------------------------
    
    for i in range(len(domain)):
        c = domain[i]
        
        # 1. 入力漏れ (Deletion)
        typo = domain[:i] + domain[i+1:]
        variants[typo][0].add("入力漏れ")

        # 2. 二重入力 (Insertion/Repetition)
        typo = domain[:i] + c + c + domain[i+1:]
        variants[typo][0].add("二重入力")

        # 3. 置換ミス (隣接キー, ホモグリフ, 左右対称, スペルミス)
        # 隣接キー誤打
        for adj in keyboard_adjacent.get(c.lower(), ''):
            typo = domain[:i] + adj + domain[i+1:]
            variants[typo][0].add("隣接キー誤打")

        # ホモグリフ・視覚類似文字
        for g in HOMOGLYPHS_FOR_GENERATOR.get(c.lower(), []):
            typo = domain[:i] + g + domain[i+1:]
            variants[typo][0].add("ホモグリフ（視覚類似文字）")
            
        # 左右対称キー誤打
        for a, b in symmetric_key_pairs:
            if c == a: variants[domain[:i] + b + domain[i+1:]][0].add("左右対称キー誤打")
            elif c == b: variants[domain[:i] + a + domain[i+1:]][0].add("左右対称キー誤打")


    # 4. 入力順序ミス (Transposition)
    for i in range(len(domain) - 1):
        swapped = domain[:i] + domain[i+1] + domain[i] + domain[i+2:]
        variants[swapped][0].add("入力順序ミス")
        
    # 5. ドット抜け (Deletion of dot)
    if '.' in domain:
        dotless = domain.replace('.', '', 1) 
        variants[dotless][0].add("ドット抜け")

    # ------------------------------------------------------------------
    # B. スコアリングとランキング
    # ------------------------------------------------------------------
    
    ranked_results = []
    
    for typo, (causes, _) in variants.items():
        if typo == domain: continue
        
        final_score = 0
        num_causes = 0
        
        # 距離1のミスの文字ペアを特定 (スコアリングの必須要件)
        c1, c2 = identify_single_replacement(domain, typo) 
        
        # 1. 基本スコアと内部重みの適用
        for cause in causes:
            W_major = major_weights.get(cause, 0)
            W_internal = 1.0 # 内部重みが見つからなかった場合のデフォルト値
            
            # ⬇️ 修正: 個別ミスの内部重み (W_internal) の参照ロジック ⬇️

            if cause in {"隣接キー誤打", "ホモグリフ（視覚類似文字）", "左右対称キー誤打", "スペルミス（認知ミス）"}:
                # 置換系 (c1, c2) の内部重みを参照
                key = (c1, c2)
                if key in internal_weights.get(cause, {}):
                    W_internal = internal_weights[cause][key]
                elif (c2, c1) in internal_weights.get(cause, {}): # 逆順もチェック
                    W_internal = internal_weights[cause][(c2, c1)]
            
            elif cause in {"入力漏れ", "ドット抜け"}:
                # 削除系 (c1, '（空）') の内部重みを参照
                key = (c1, '（空）') 
                if key in internal_weights.get(cause, {}):
                    W_internal = internal_weights[cause][key]
            
            elif cause == "二重入力":
                # 挿入系 ('（空）', c2) の内部重みをを参照
                key = ('（空）', c2) 
                if key in internal_weights.get(cause, {}):
                    W_internal = internal_weights[cause][key]

            # スコアの計算: W_大 * W_内
            final_score += W_major * W_internal
            num_causes += 1 

        # 2. 複合ミス ペナルティの適用 (DL距離 > 1 のミスを抑制)
        if num_causes > 1:
            COMPOSITE_PENALTY_FACTOR = 0.5 
            final_score *= COMPOSITE_PENALTY_FACTOR

        # 3. ❌ 修正: 入力漏れ/二重入力 ボーナスの削除 ❌
        # if "入力漏れ" in causes or "二重入力" in causes:
        #     final_score += 0.05 
            
        distance = damerau_levenshtein_distance(domain, typo)
        
        ranked_results.append({
            "typo": typo,
            "causes": '・'.join(sorted(causes)),
            "score": round(final_score, 3),
            "distance": distance
        })

    # スコア降順、距離昇順でランキング
    ranked_results.sort(key=lambda x: (x['score'], -x['distance']), reverse=True)

    return ranked_results[:top_n]


# ==============================================================================
# 🚀 6. メイン実行ブロック (変更なし)
# ==============================================================================

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # 必須ファイルパス設定
    INPUT_FILE = "filtered_address.csv"
    DL4_FILTERED_FILE = "filtered_domain_typos_dl4.csv"
    CAUSES_CSV_FILE = "domaintypos_dl4_causes2.csv"
    DL_THRESHOLD = 4

    # 1. タイポの抽出とフィルタリング
    filter_domain_differences_with_mismatch(INPUT_FILE, DL4_FILTERED_FILE, DL_THRESHOLD)

    # 2. 原因分類を付与しCSVに出力
    append_typo_causes(DL4_FILTERED_FILE, CAUSES_CSV_FILE)
    
    # 3. 原因別集計と詳細な分析レポート (重み計算を含む)
    # 戻り値: (大カテゴリ重み, 個別ミス内部重み)
    major_weights, internal_weights = analyze_and_report_all(CAUSES_CSV_FILE)
    
    # 4. ドメインランキング生成
    correct_domain = input("\n正しいドメイン名を入力してください（例: treasurefactory.co.jp）: ").strip()
    
    if major_weights:
        print("\n" + "=" * 78)
        print(f"🥇 '{correct_domain}' に対する予測タイポドメインランキング:\n")
        
        predicted_typos = typo_generator_ranked(correct_domain, major_weights, internal_weights, top_n=10)
        
        for i, r in enumerate(predicted_typos):
            print(f"{i+1}位 {r['typo']:<30} (スコア: {r['score']:.3f}, 距離: {r['distance']}, 原因: {r['causes']})")
    else:
        print("\n[エラー] 重みデータが計算されていないため、ランキング生成を実行できませんでした。")
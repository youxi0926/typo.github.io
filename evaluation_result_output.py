import pandas as pd
import numpy as np
import Levenshtein
from pyxdameraulevenshtein import damerau_levenshtein_distance
from sklearn.model_selection import KFold
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import difflib
import time

# ==============================================================================
# 1. 定数とキーボード設定
# ==============================================================================

KEYBOARD_ADJACENT = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qws', 's': 'qwedazx', 'd': 'erfcsx', 'f': 'rtdgvcj',
    'g': 'tyfhvbn', 'h': 'yugjnb', 'j': 'uikhmnf', 'k': 'ijolm', 'l': 'okp',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk,',
    '.': ',/', ',': 'm.', '-': '^'
}

SYMMETRIC_KEY_PAIRS = [('f', 'j'), ('d', 'k'), ('s', 'l'), ('a', ';')]
HOMOGLYPHS = {'1': ['l'], '0': ['o'], 'i': ['l'], 'l': ['i'], 'r': ['m'], 'b': ['d'], 'd': ['b']}
EMPTY_CHAR = '（空）'

# ==============================================================================
# 2. ヘルパー関数
# ==============================================================================

def extract_domain(email):
    return str(email).split('@')[-1] if '@' in str(email) else str(email)

def identify_single_replacement(correct, typo):
    matcher = SequenceMatcher(None, correct, typo)
    ops = matcher.get_opcodes()
    repls = [(correct[i1:i2], typo[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag == 'replace']
    ins = [typo[j1:j2] for tag, i1, i2, j1, j2 in ops if tag == 'insert']
    dels = [correct[i1:i2] for tag, i1, i2, j1, j2 in ops if tag == 'delete']
    
    if len(repls) == 1 and not ins and not dels:
        if len(repls[0][0]) == 1 and len(repls[0][1]) == 1: return repls[0]
    if len(ins) == 1 and not repls and not dels: return (EMPTY_CHAR, ins[0][0])
    if len(dels) == 1 and not repls and not ins: return (dels[0][0], EMPTY_CHAR)
    return ('', '')

def get_transposed_pair(correct, typo):
    if damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1:
        for i in range(len(correct) - 1):
            if correct[i] == typo[i+1] and correct[i+1] == typo[i]:
                return (correct[i], correct[i+1])
    return None

def extract_ngram_diffs(correct, typo):
    sm = difflib.SequenceMatcher(None, correct, typo)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal': continue
        src = correct[i1:i2] or EMPTY_CHAR
        tgt = typo[j1:j2] or EMPTY_CHAR
        diffs.append((src, tgt))
    return diffs

# ==============================================================================
# 3. 学習・生成ロジック
# ==============================================================================

def train_model(train_df):
    cause_diff_counter = defaultdict(Counter)
    total_samples = len(train_df)

    for _, row in train_df.iterrows():
        correct = extract_domain(row['correct_address'])
        typo = extract_domain(row['input_address'])
        diffs = extract_ngram_diffs(correct, typo)
        
        t_pair = get_transposed_pair(correct, typo)
        if t_pair:
            key = f"{t_pair[0]} {t_pair[1]} -> {t_pair[1]} {t_pair[0]}"
            cause_diff_counter["入力順序ミス"][key] += 1
            continue

        for c1, c2 in diffs:
            if c1 != EMPTY_CHAR and c2 != EMPTY_CHAR:
                cause_diff_counter["置換系"][(c1, c2)] += 1
            elif c1 == EMPTY_CHAR:
                cause_diff_counter["二重入力"][(c1, c2)] += 1
            else:
                cause_diff_counter["入力漏れ"][(c1, c2)] += 1

    weights = {}
    for cause, counter in cause_diff_counter.items():
        weights[cause] = {k: v / total_samples for k, v in counter.items()}
    return weights

def generate_and_rank(domain, weights):
    variants = defaultdict(set)
    for i in range(len(domain)):
        c = domain[i].lower()
        variants[domain[:i] + domain[i+1:]].add("入力漏れ" if c != '.' else "ドット抜け")
        variants[domain[:i] + c + c + domain[i+1:]].add("二重入力")
        if c in KEYBOARD_ADJACENT:
            for adj in KEYBOARD_ADJACENT[c]:
                variants[domain[:i] + adj + domain[i+1:]].add("置換系")
        if c in HOMOGLYPHS:
            for g in HOMOGLYPHS[c]:
                variants[domain[:i] + g + domain[i+1:]].add("置換系")
        if i < len(domain) - 1:
            variants[domain[:i] + domain[i+1] + domain[i] + domain[i+2:]].add("入力順序ミス")

    results = []
    for typo, causes in variants.items():
        if typo == domain: continue
        score = 0.0
        c1, c2 = identify_single_replacement(domain, typo)
        for cause in causes:
            w_cause = "置換系" if cause in ["隣接キー誤打", "左右対称キー誤打", "ホモグリフ（視覚類似文字）", "置換系"] else cause
            if cause == "入力順序ミス":
                t_pair = get_transposed_pair(domain, typo)
                if t_pair:
                    key = f"{t_pair[0]} {t_pair[1]} -> {t_pair[1]} {t_pair[0]}"
                    score += weights.get("入力順序ミス", {}).get(key, 0.0)
            else:
                score += weights.get(w_cause, {}).get((c1, c2), 0.0)
                if score == 0 and c1 != EMPTY_CHAR and c2 != EMPTY_CHAR:
                    score += weights.get(w_cause, {}).get((c2, c1), 0.0)
        
        if len(causes) > 1: score *= 0.5
        score += 0.000001
        results.append({"typo": typo, "score": score, "distance": damerau_levenshtein_distance(domain, typo)})
    results.sort(key=lambda x: (-x['score'], x['distance']))
    return results

# ==============================================================================
# 4. メイン評価関数（サンプル表示機能追加）
# ==============================================================================

def main_evaluation(csv_path):
    print(f"--- タイポ生成モデル評価システム (サンプル出力モード) ---")
    start_time = time.time()
    
    df = pd.read_csv(csv_path)
    df = df[df.apply(lambda r: damerau_levenshtein_distance(extract_domain(r['correct_address']), extract_domain(r['input_address'])) <= 1, axis=1)]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_fold_gen_counts = []
    k_values = [1, 10, 30, 50, 100, 200]
    all_fold_top_k = {k: [] for k in k_values}

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        model_weights = train_model(train_df)
        
        fold_gen_counts = []
        fold_hits = {k: 0 for k in k_values}
        
        print(f"\n[Fold {fold+1}] 評価中...")
        
        # サンプル出力用のカウンタ
        samples_shown = 0
        MAX_SAMPLES = 2 # 各Foldで2件だけ詳細を表示
        
        for _, row in test_df.iterrows():
            correct = extract_domain(row['correct_address'])
            actual = extract_domain(row['input_address'])
            predictions = generate_and_rank(correct, model_weights)
            
            fold_gen_counts.append(len(predictions))
            pred_list = [p['typo'] for p in predictions]

            # サンプル表示
            if samples_shown < MAX_SAMPLES:
                print("-" * 50)
                print(f"サンプル {samples_shown + 1}:")
                print(f"  正解ドメイン: {correct}")
                print(f"  実際のミス  : {actual}")
                print(f"  モデル予測 (Top-10):")
                for i, p in enumerate(predictions[:10]):
                    hit_mark = "★HIT" if p['typo'] == actual else ""
                    print(f"    {i+1:>2}位: {p['typo']:<25} (Score: {p['score']:.5f}) {hit_mark}")
                samples_shown += 1

            for k in k_values:
                if actual in pred_list[:k]:
                    fold_hits[k] += 1
        
        avg_gen = np.mean(fold_gen_counts)
        all_fold_gen_counts.append(avg_gen)
        for k in k_values:
            all_fold_top_k[k].append(fold_hits[k] / len(test_df))

    # --- 最終レポート ---
    end_time = time.time()
    print("\n" + "="*60)
    print("■ 最終評価レポート")
    print("="*60)
    print(f"平均生成数: {np.mean(all_fold_gen_counts):.2f} 件")
    print("-" * 60)
    for k in k_values:
        avg_acc = np.mean(all_fold_top_k[k]) * 100
        print(f"k={k:<3} | 平均網羅率: {avg_acc:>6.2f} %")
    print("="*60)

if __name__ == "__main__":
    main_evaluation("filtered_address.csv")
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
# 1. 定数と環境設定
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
# 2. 生成エンジン (W_individualベース)
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

def generate_typo_squatting_candidates(domain, weights):
    """
    提案モデルに基づきタイポ候補を全生成し、スコア順にソートしたリストを返す。
    """
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
        
        score = score * 0.5 if len(causes) > 1 else score
        score += 0.000001
        results.append({"typo": typo, "score": score})

    results.sort(key=lambda x: x['score'], reverse=True)
    return [r['typo'] for r in results]

# ==============================================================================
# 3. 学習エンジン
# ==============================================================================

def train_individual_weights(train_df):
    cause_diff_counter = defaultdict(Counter)
    total_samples = len(train_df)

    for _, row in train_df.iterrows():
        correct = extract_domain(row['correct_address'])
        typo = extract_domain(row['input_address'])
        
        t_pair = get_transposed_pair(correct, typo)
        if t_pair:
            key = f"{t_pair[0]} {t_pair[1]} -> {t_pair[1]} {t_pair[0]}"
            cause_diff_counter["入力順序ミス"][key] += 1
            continue
        
        sm = difflib.SequenceMatcher(None, correct, typo)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal': continue
            c1 = correct[i1:i2] or EMPTY_CHAR
            c2 = typo[j1:j2] or EMPTY_CHAR
            if c1 != EMPTY_CHAR and c2 != EMPTY_CHAR:
                cause_diff_counter["置換系"][(c1, c2)] += 1
            elif c1 == EMPTY_CHAR:
                cause_diff_counter["二重入力"][(c1, c2)] += 1
            else:
                cause_diff_counter["入力漏れ"][(c1, c2)] += 1

    weights = {cause: {k: v / total_samples for k, v in counter.items()} for cause, counter in cause_diff_counter.items()}
    return weights

# ==============================================================================
# 4. 評価エンジン
# ==============================================================================

def evaluate_fold(test_data, train_weights, k_values):
    results = {k: 0 for k in k_values}
    total = len(test_data)
    generated_domains_size = []

    for _, row in test_data.iterrows():
        input_domain = extract_domain(row['correct_address'])
        target_domain = extract_domain(row['input_address'])

        generated_domains = generate_typo_squatting_candidates(input_domain, train_weights)
        generated_domains_size.append(len(generated_domains))

        for k in k_values:
            if target_domain in generated_domains[:k]:
                results[k] += 1

    accuracy_results = {k: count / total for k, count in results.items()}
    avg_gen = sum(generated_domains_size) / len(generated_domains_size)
    hit_30 = results.get(31, results.get(30, 0)) / total # 網羅率表示用

    return accuracy_results, avg_gen, hit_30

# ==============================================================================
# 5. メイン実行 (5-Fold Cross Validation)
# ==============================================================================

def main():
    print("--- タイポ生成モデル評価システム 起動 ---")
    start_time = time.time()
    
    csv_path = "filtered_address.csv"
    try:
        df_raw = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: {csv_path} が見つかりません。")
        return

    # 距離1限定かつ距離0を除外
    df = df_raw[df_raw.apply(lambda r: damerau_levenshtein_distance(extract_domain(r['correct_address']), extract_domain(r['input_address'])) == 1, axis=1)].copy()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k_values = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141, 146,
 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 200]
    
    all_fold_accuracies = {k: [] for k in k_values}
    all_fold_gen_counts = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        train_weights = train_individual_weights(train_df)
        
        fold_acc, fold_gen, hit_30 = evaluate_fold(test_df, train_weights, k_values)
        
        all_fold_gen_counts.append(fold_gen)
        for k in k_values:
            all_fold_accuracies[k].append(fold_acc[k])
            
        print(f"Fold {fold + 1} 完了: 平均生成数 {fold_gen:.2f}件 | Top-30網羅率 {hit_30:.2%}")

    processing_time = time.time() - start_time

    # ==============================================================================
    # 最終レポート出力
    # ==============================================================================
    print("\n" + "="*60)
    print("■ 8.3 評価結果（最終レポート）")
    print("="*60)
    print(f"評価対象データ総数: {len(df)} 件")
    print(f"処理時間: {processing_time:.2f} 秒")
    print("-" * 60)
    
    print("(1) タイポスクワッティングドメインの生成数")
    for i, count in enumerate(all_fold_gen_counts):
        print(f"    Fold {i+1}: {count:.2f} 件")
    print(f"    👉 5つのテストデータの平均生成数: {np.mean(all_fold_gen_counts):.2f} 件")
    
    print("-" * 60)
    
    print("(2) 生成モデルの網羅率 (Top-k Accuracy)")
    print("    k値   |  平均網羅率 (Accuracy)")
    print("----------|----------------------")
    for k in k_values:
        mean_val = np.mean(all_fold_accuracies[k]) * 100
        std_val = np.std(all_fold_accuracies[k]) * 100
        print(f"    k={k:<3} |   {mean_val:>5.2f} %  (±{std_val:.2f})")

    print("\n[考察用アドバイス]")
    max_acc = np.mean(all_fold_accuracies[200]) * 100
    print(f"・あなたのモデルの最大網羅率は {max_acc:.2f}% です。")
    print("・先輩の25%と比較して、どれだけ向上したか数値で示せます。")
    print("・データ量が多いため、k=30付近での収束がより顕著に見えるはずです。")
    print("="*60)

if __name__ == "__main__":
    main()
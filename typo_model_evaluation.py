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
# 1. 定数とキーボード設定（あなたの研究の定義を反映）
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
# 2. ヘルパー関数（抽出・分類・特定）
# ==============================================================================

def extract_domain(email):
    """アドレスからドメイン部を抽出"""
    return str(email).split('@')[-1] if '@' in str(email) else str(email)

def identify_single_replacement(correct, typo):
    """DL距離1の差分ペア(c1, c2)を特定する"""
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
    """転置ミス（入力順序ミス）の文字ペアを抽出"""
    if damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1:
        for i in range(len(correct) - 1):
            if correct[i] == typo[i+1] and correct[i+1] == typo[i]:
                return (correct[i], correct[i+1])
    return None

def extract_ngram_diffs(correct, typo):
    """全ての差分ペアを抽出（学習用）"""
    sm = difflib.SequenceMatcher(None, correct, typo)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal': continue
        src = correct[i1:i2] or EMPTY_CHAR
        tgt = typo[j1:j2] or EMPTY_CHAR
        diffs.append((src, tgt))
    return diffs

# ==============================================================================
# 3. 学習・生成ロジックの統合
# ==============================================================================

def train_model(train_df):
    """トレーニングデータから個別ミスの発生確率(W_individual)を計算"""
    cause_diff_counter = defaultdict(Counter)
    total_samples = len(train_df)

    for _, row in train_df.iterrows():
        correct = extract_domain(row['correct_address'])
        typo = extract_domain(row['input_address'])
        
        # 簡易的な原因分類（学習用）
        diffs = extract_ngram_diffs(correct, typo)
        
        # 転置の特別処理
        t_pair = get_transposed_pair(correct, typo)
        if t_pair:
            key = f"{t_pair[0]} {t_pair[1]} -> {t_pair[1]} {t_pair[0]}"
            cause_diff_counter["入力順序ミス"][key] += 1
            continue

        for c1, c2 in diffs:
            # 置換、挿入、削除の重みを集計
            if c1 != EMPTY_CHAR and c2 != EMPTY_CHAR:
                cause_diff_counter["置換系"][(c1, c2)] += 1
            elif c1 == EMPTY_CHAR:
                cause_diff_counter["二重入力"][(c1, c2)] += 1
            else:
                cause_diff_counter["入力漏れ"][(c1, c2)] += 1

    # 重みの正規化 (W_individual = 特定ミスの件数 / 全タイポ件数)
    weights = {}
    for cause, counter in cause_diff_counter.items():
        weights[cause] = {k: v / total_samples for k, v in counter.items()}
    
    return weights

def generate_and_rank(domain, weights):
    """学習した重みに基づき、指定ドメインのタイポ候補を生成・ソートする"""
    variants = defaultdict(set) # {typo: {causes}}
    
    # --- 生成ルール ---
    for i in range(len(domain)):
        c = domain[i].lower()
        # 1. 削除 (入力漏れ / ドット抜け)
        variants[domain[:i] + domain[i+1:]].add("入力漏れ" if c != '.' else "ドット抜け")
        # 2. 二重入力
        variants[domain[:i] + c + c + domain[i+1:]].add("二重入力")
        # 3. 隣接キー
        if c in KEYBOARD_ADJACENT:
            for adj in KEYBOARD_ADJACENT[c]:
                variants[domain[:i] + adj + domain[i+1:]].add("置換系")
        # 4. ホモグリフ
        if c in HOMOGLYPHS:
            for g in HOMOGLYPHS[c]:
                variants[domain[:i] + g + domain[i+1:]].add("置換系")
        # 5. 転置
        if i < len(domain) - 1:
            variants[domain[:i] + domain[i+1] + domain[i] + domain[i+2:]].add("入力順序ミス")

    # --- スコアリング ---
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
                # 逆順チェック (隣接ミスなどは逆方向もあり得るため)
                if score == 0 and c1 != EMPTY_CHAR and c2 != EMPTY_CHAR:
                    score += weights.get(w_cause, {}).get((c2, c1), 0.0)
        
        # 複合ミスペナルティ
        if len(causes) > 1: score *= 0.5
        # 微小なボーナス（生成されたこと自体への評価）
        score += 0.000001
        
        results.append({
            "typo": typo,
            "score": score,
            "distance": damerau_levenshtein_distance(domain, typo)
        })

    # スコアが高い順、編集距離が短い順にソート
    results.sort(key=lambda x: (-x['score'], x['distance']))
    return results

# ==============================================================================
# 4. 5-分割交差検証の実行と集計
# ==============================================================================

def main_evaluation(csv_path):
    print(f"--- タイポ生成モデル評価システム 起動 ---")
    start_time = time.time()
    
    df = pd.read_csv(csv_path)
    # 先輩の基準（DL距離1）で評価。2以上も含めるならここを調整。
    df = df[df.apply(lambda r: damerau_levenshtein_distance(extract_domain(r['correct_address']), extract_domain(r['input_address'])) <= 1, axis=1)]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_fold_gen_counts = []
    k_values = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141, 146,
 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 200]
    all_fold_top_k = {k: [] for k in k_values}

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        
        # 学習
        model_weights = train_model(train_df)
        
        fold_gen_counts = []
        fold_hits = {k: 0 for k in k_values}
        
        # 評価
        for _, row in test_df.iterrows():
            correct = extract_domain(row['correct_address'])
            actual = extract_domain(row['input_address'])
            
            # 生成
            predictions = generate_and_rank(correct, model_weights)
            fold_gen_counts.append(len(predictions))
            
            # Top-k 照合
            pred_list = [p['typo'] for p in predictions]
            for k in k_values:
                if actual in pred_list[:k]:
                    fold_hits[k] += 1
        
        # 集計
        avg_gen = np.mean(fold_gen_counts)
        all_fold_gen_counts.append(avg_gen)
        
        print(f"Fold {fold+1} 完了: 平均生成数 {avg_gen:.2f}件 | Top-30網羅率 {(fold_hits[31]/len(test_df))*100:.2f}%")
        
        for k in k_values:
            all_fold_top_k[k].append(fold_hits[k] / len(test_df))

    # ==============================================================================
    # 5. レポート出力（先輩の論文形式）
    # ==============================================================================
    end_time = time.time()
    
    print("\n" + "="*60)
    print("■ 8.3 評価結果（最終レポート）")
    print("="*60)
    print(f"評価対象データ総数: {len(df)} 件")
    print(f"処理時間: {end_time - start_time:.2f} 秒")
    print("-" * 60)
    
    # (1) 生成数の評価
    print(f"(1) タイポスクワッティングドメインの生成数")
    for i, count in enumerate(all_fold_gen_counts):
        print(f"    Fold {i+1}: {count:.2f} 件")
    print(f"    👉 5つのテストデータの平均生成数: {np.mean(all_fold_gen_counts):.2f} 件")
    
    print("-" * 60)
    
    # (2) 網羅率の評価
    print(f"(2) 生成モデルの網羅率 (Top-k Accuracy)")
    print(f"    k値   |  平均網羅率 (Accuracy)")
    print(f"----------|----------------------")
    for k in k_values:
        avg_acc = np.mean(all_fold_top_k[k]) * 100
        std_acc = np.std(all_fold_top_k[k]) * 100
        print(f"    k={k:<3} |  {avg_acc:>6.2f} %  (±{std_acc:.2f})")
    
    print("\n[考察用アドバイス]")
    max_acc = np.mean(all_fold_top_k[max(k_values)]) * 100
    print(f"・あなたのモデルの最大網羅率は {max_acc:.2f}% です。")
    print(f"・先輩の25%と比較して、どれだけ向上したか数値で示せます。")
    print(f"・データ量が多いため、k=30付近での収束がより顕著に見えるはずです。")
    print("="*60)

if __name__ == "__main__":
    # ファイル名が異なる場合はここを修正してください
    main_evaluation("filtered_address.csv")

import os
import re
import pandas as pd
from difflib import SequenceMatcher
from collections import defaultdict, Counter
from pyxdameraulevenshtein import damerau_levenshtein_distance
import Levenshtein

# ==============================================================================
# 🗄️ 1. 設定と定数
# ==============================================================================

# QWERTYキーボード隣接キーマップ（より網羅的で実用的なものに修正・簡略化）
KEYBOARD_ADJACENT = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qwsz', 's': 'qwedazx', 'd': 'erfcsx', 'f': 'rtdgvc',
    'g': 'tyfhvb', 'h': 'yugjnb', 'j': 'uikhm', 'k': 'ijolm', 'l': 'okp',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
    '.': ',/', ',': 'm.', '/': '.',
    '1': 'q2', '2': 'qwe3', '3': 'ert4', '4': 'rty5', '5': 'tyu6',
    '6': 'yui7', '7': 'uio8', '8': 'iop9', '9': 'op0', '0': 'p-9'
}

# 左右対称配置キー誤打（例: f ↔ j）
SYMMETRIC_KEY_PAIRS = [('f', 'j'), ('d', 'k'), ('s', 'l'), ('a', ';')]

# ホモグリフ（視覚類似文字）
HOMOGLYPH_PAIRS = [('1', 'l'), ('0', 'o'), ('i', 'l'), ('rn', 'm'), ('а', 'a')]
HOMOGLYPHS = {'1': ['l'], '0': ['o'], 'i': ['l'], 'r': ['m', 'n'], 'a': ['а']} # 生成用に調整

# タイポ原因の重み（スコアリング用）
TYPO_WEIGHTS = {
    "スペルミス（認知ミス）": 0.285, "二重入力": 0.209, "入力漏れ": 0.203,
    "隣接キー誤打": 0.162, "別TLD結合": 0.058, "ホモグリフ誤認": 0.022,
    "別LD結合・ドット抜け": 0.042, # ドット抜けと別LD結合を統合
    "入力順序ミス": 0.019, "左右対称キー誤打": 0.001,
}

# ==============================================================================
# ⚙️ 2. ヘルパー関数
# ==============================================================================

def extract_domain(email: str) -> str:
    """メールアドレスからドメイン部を抽出します。"""
    return email.split('@')[1] if '@' in email else ''

def keyboard_adjacent_check(c1: str, c2: str) -> bool:
    """文字 c1 と c2 がキーボードで隣接しているかを確認します。"""
    return c1.lower() in KEYBOARD_ADJACENT and c2.lower() in KEYBOARD_ADJACENT[c1.lower()]

def is_symmetric_mismatch(c1: str, c2: str) -> bool:
    """文字 c1 と c2 が左右対称キーの誤打であるかを確認します。"""
    return any((c1 == a and c2 == b) or (c1 == b and c2 == a) for a, b in SYMMETRIC_KEY_PAIRS)

def is_visual_homoglyph(c1: str, c2: str) -> bool:
    """文字 c1 と c2 が視覚的に類似しているか（ホモグリフ）を確認します。"""
    return any((c1 == a and c2 == b) or (c1 == b and c2 == a) for a, b in HOMOGLYPH_PAIRS)

def is_valid_tld(tld: str) -> bool:
    """TLDが有効な形式か（アルファベットのみで2〜4文字）をチェックします。"""
    return len(tld) in [2, 3, 4] and tld.isalpha()

# ==============================================================================
# 🔬 3. タイポ分類ロジック
# ==============================================================================

def classify_edit_ops_japanese(correct: str, typo: str) -> dict:
    """
    正しいドメインと誤ったドメイン間の編集操作を分析し、タイポ原因を分類します。
    （元のコードのロジックを踏襲）
    """
    causes = set()
    correct_parts = []
    typo_parts = []

    # Damerau-Levenshtein距離1かつLevenshtein距離>1は、単独の転置と判断（優先）
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
            elif is_visual_homoglyph(c1, c2): causes.add("ホモグリフ誤認")
            else: causes.add("スペルミス（認知ミス）")

        elif tag == 'insert':
            correct_parts.append(''); typo_parts.append(c2)
            # TLD関連の特殊な挿入
            if correct.endswith('.co') and typo.endswith('.co.jp'): causes.add("別TLD結合")
            elif correct.endswith('.co.jp') and typo.endswith('cojp') and len(ops) == 1: causes.add("別LD結合・ドット抜け") # ドット抜けが挿入と判定されるケース
            else: causes.add("二重入力")

        elif tag == 'delete':
            correct_parts.append(c1); typo_parts.append('')
            causes.add("入力漏れ")

    # TLD違いの詳細判定（末尾のみを比較）
    correct_parts_tld = correct.split('.')
    typo_parts_tld = typo.split('.')
    if len(correct_parts_tld) > 1 and len(typo_parts_tld) > 1 and correct_parts_tld[-1] != typo_parts_tld[-1]:
        correct_tld = correct_parts_tld[-1]
        typo_tld = typo_parts_tld[-1]
        if is_valid_tld(typo_tld):
            if damerau_levenshtein_distance(correct_tld, typo_tld) == 1:
                # TLD間の置換が隣接キー誤打である可能性
                if all(keyboard_adjacent_check(c1, c2) for c1, c2 in zip(correct_tld, typo_tld)):
                    causes.add("隣接キー誤打") # TLD内の隣接キー誤打
                else:
                    causes.add("別TLD結合") # 誤ったTLDを入力した（認知ミス）

    # ドット抜け（別LD結合）判定
    if '.' in correct and '.' not in typo and damerau_levenshtein_distance(correct.replace('.', ''), typo) == 0:
        causes.add("別LD結合・ドット抜け")

    if not causes and correct != typo:
        # TLD違いなどでEditOpsで捉えきれないが、Damerau-Levenshtein距離>0の場合のフォールバック
        causes.add("スペルミス（認知ミス）")

    return {
        "cause": '・'.join(sorted(causes)) if causes else "一致",
        "correct_part": ' '.join(correct_parts),
        "mismatched_part": ' '.join(typo_parts)
    }

# ==============================================================================
# 🛠️ 4. タイポドメイン生成ロジック
# ==============================================================================

class TypoGenerator:
    """
    タイポ原因の重み付けに基づいて、可能性の高いタイポドメインを生成・ランキング付けします。
    """
    def __init__(self, typo_weights: dict):
        self.typo_weights = typo_weights

    def generate_variants(self, domain: str) -> list:
        """各種ルールに基づき、ドメインのタイポ候補を生成します。"""
        variants = []

        # 1. 一文字置換、挿入、削除（距離1の操作）
        for i in range(len(domain)):
            c = domain[i]

            # 入力漏れ（削除）
            variants.append((domain[:i] + domain[i+1:], ["入力漏れ"]))

            # 二重入力（挿入: 2文字入力）
            variants.append((domain[:i] + c + c + domain[i+1:], ["二重入力"]))

            # 隣接キー誤打（置換）
            for adj in KEYBOARD_ADJACENT.get(c.lower(), ''):
                variants.append((domain[:i] + adj + domain[i+1:], ["隣接キー誤打"]))

            # ホモグリフ誤認（置換）
            for base, glyphs in HOMOGLYPHS.items():
                if c == base:
                    for g in glyphs: variants.append((domain[:i] + g + domain[i+1:], ["ホモグリフ誤認"]))
                elif c in glyphs:
                    variants.append((domain[:i] + base + domain[i+1:], ["ホモグリフ誤認"]))

            # 左右対称キー誤打（置換）
            for a, b in SYMMETRIC_KEY_PAIRS:
                if c == a: variants.append((domain[:i] + b + domain[i+1:], ["左右対称キー誤打"]))
                elif c == b: variants.append((domain[:i] + a + domain[i+1:], ["左右対称キー誤打"]))

        # 2. 入力順序ミス（隣接文字の転置）
        for i in range(len(domain) - 1):
            swapped = domain[:i] + domain[i+1] + domain[i] + domain[i+2:]
            variants.append((swapped, ["入力順序ミス"]))

        # 3. TLD関連の特殊ルール
        # 別TLD結合
        tld_map = {".co.jp": ".com", ".com": ".co.jp", ".net": ".com"}
        for from_tld, to_tld in tld_map.items():
            if domain.endswith(from_tld):
                variants.append((domain.replace(from_tld, to_tld), ["別TLD結合"]))

        # ドット抜け／別LD結合
        if '.' in domain:
            dotless = domain.replace('.', '')
            # ドット抜けはドメイン全体で1回のみ
            if damerau_levenshtein_distance(domain, dotless) == domain.count('.'):
                variants.append((dotless, ["別LD結合・ドット抜け"]))

        return variants

    def rank_typos(self, domain: str, max_distance: int = 2, top_n: int = 30) -> list:
        """生成されたタイポ候補を重み付けスコアでランキングします。"""
        variants = self.generate_variants(domain)
        seen = {} # {typo: (causes, score)}

        for typo, causes in variants:
            if typo == domain: continue
            if damerau_levenshtein_distance(domain, typo) > max_distance: continue

            # 重み付けスコアの計算（原因の合計スコア）
            score = sum(self.typo_weights.get(c, 0) for c in causes)

            # より高いスコアの原因セットで更新
            if typo not in seen or score > seen[typo][1]:
                # causesをユニークでソートされたリストにする（元のコードに倣い）
                sorted_causes = sorted(list(set(causes)))
                seen[typo] = (sorted_causes, score)

        ranked = sorted(seen.items(), key=lambda x: (x[1][1], x[0]), reverse=True) # スコア降順、ドメイン名昇順

        return [{
            "typo": typo,
            "causes": "・".join(causes),
            "score": f"{score:.3f}"
        } for typo, (causes, score) in ranked[:top_n]]

# ==============================================================================
# 🚀 5. 実行例
# ==============================================================================

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # 実行例 1: タイポドメインのランキング生成
    # --------------------------------------------------------------------------
    print("------------------------------------------")
    print("📝 タイポドメイン候補の生成とランキング")
    print("------------------------------------------")
    target_domain = "treasurefactory.co.jp"
    typo_gen = TypoGenerator(TYPO_WEIGHTS)

    results = typo_gen.rank_typos(target_domain, max_distance=1) # 距離1に限定して生成
    print(f"\n✅ 元ドメイン: '{target_domain}' に対するタイポランキング（DL距離 ≦ 1）:\n")
    for i, r in enumerate(results):
        print(f"{i+1}位 {r['typo']:<30} (スコア: {r['score']}, 原因: {r['causes']})")
    print()

    # --------------------------------------------------------------------------
    # 実行例 2: 原因別集計と割合（重み）の表示
    # --------------------------------------------------------------------------
    # ※ この部分は、データフレームのファイル（domaintypos_dl4_causes2.csv）が存在しないと実行できませんが、
    #     コードの最終ブロックにある集計ロジックを再利用できます。

    print("------------------------------------------")
    print("📊 原因別集計と割合（重み）")
    print("------------------------------------------")
    print("\n■ 設定されている原因別集計と割合（重み）:\n")
    total = sum(TYPO_WEIGHTS.values())
    for cause in sorted(TYPO_WEIGHTS, key=TYPO_WEIGHTS.get, reverse=True):
        weight = TYPO_WEIGHTS[cause]
        print(f"{cause:<25}: ({weight/total*100:6.2f}%) 重み: {weight:>5.3f}")

    # --------------------------------------------------------------------------
    # 実行例 3: 差分分析（CSVファイルが存在する場合のみ有効）
    # --------------------------------------------------------------------------
    if os.path.exists("domaintypos_dl4_causes2.csv"):
        print("\n------------------------------------------")
        print("🔍 原因別Nグラム差分分析")
        print("------------------------------------------")
        analyze_ngram_differences("domaintypos_dl4_causes2.csv")
    else:
        print("\n[注意] 'domaintypos_dl4_causes2.csv' が見つからないため、差分分析はスキップします。")




# ------タイポドメイン生成器---------
def typo_generator(domain: str, max_distance: int = 2):
    typo_candidates = set()

    # 各原因ルールでタイポ生成
    typo_candidates |= generate_adjacent_key_typos(domain)       # 隣接キー誤打
    typo_candidates |= generate_omission_typos(domain)           # 入力漏れ
    typo_candidates |= generate_double_typos(domain)             # 二重入力
    typo_candidates |= generate_order_swap_typos(domain)         # 入力順序
    typo_candidates |= generate_visual_confusion_typos(domain)   # ホモグリフ
    typo_candidates |= generate_tld_typos(domain)                # TLD変形
    typo_candidates |= generate_symmetric_key_typos(domain)      # 左右対称
    typo_candidates |= generate_dot_missing_typos(domain)        # ドット忘れ

    # Levenshtein距離でフィルタリング
    typo_candidates = {
        typo for typo in typo_candidates
        if levenshtein_distance(domain, typo) <= max_distance and typo != domain
    }

    return list(typo_candidates)

# 原因別に差分パターンをCSV出力
def analyze_and_export_ngram_diffs(csv_path, output_dir="ngram_diff_output"):
    df = pd.read_csv(csv_path)
    cause_diff_counter = defaultdict(Counter)

    for _, row in df.iterrows():
        correct = extract_domain(str(row['correct_address']))
        typo = extract_domain(str(row['input_address']))
        cause_field = str(row['cause'])
        causes = [c.strip() for c in cause_field.split('・')]
        diffs = extract_ngram_diffs(correct, typo)

        for cause in causes:
            for c1, c2 in diffs:
                cause_diff_counter[cause][(c1, c2)] += 1

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # # 原因ごとに CSV 保存
    # for cause, counter in cause_diff_counter.items():
    #     data = [{
    #         'correct_part': c1,
    #         'typo_part': c2,
    #         'count': count
    #     } for (c1, c2), count in counter.most_common()]

    #     output_path = os.path.join(output_dir, f"ngram_diffs_{cause}.csv")
    #     df_out = pd.DataFrame(data)
    #     df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    #     print(f"→ {output_path} に出力しました")

# 実行（ファイル名を調整してください）
analyze_and_export_ngram_diffs("domaintypos_dl4_causes2.csv")








消去予定

def typo_generator(domain):
    typo_variants = []

    # ドメイン分割
    match = re.match(r'^(.+?)\.(.+)$', domain)
    if not match:
        return []

    local, tld = match.groups()

    # 隣接キー誤打
    for i, c in enumerate(local):
        if c in keyboard_adjacent:
            for adj in keyboard_adjacent[c]:
                typo = local[:i] + adj + local[i+1:]
                typo_variants.append((f"{typo}.{tld}", "隣接キー誤打"))

    # 入力漏れ
    for i in range(len(local)):
        typo = local[:i] + local[i+1:]
        typo_variants.append((f"{typo}.{tld}", "入力漏れ"))

    # 二重入力
    for i in range(len(local)):
        typo = local[:i] + local[i] + local[i:]
        typo_variants.append((f"{typo}.{tld}", "二重入力"))

    # 入力順序ミス（転置）
    for i in range(len(local) - 1):
        typo = local[:i] + local[i+1] + local[i] + local[i+2:]
        typo_variants.append((f"{typo}.{tld}", "入力順序ミス"))

    # ホモグリフ置換
    for i, c in enumerate(local):
        for a, b in homoglyph_pairs:
            if c == a:
                typo = local[:i] + b + local[i+1:]
                typo_variants.append((f"{typo}.{tld}", "ホモグリフ誤認"))
            elif c == b:
                typo = local[:i] + a + local[i+1:]
                typo_variants.append((f"{typo}.{tld}", "ホモグリフ誤認"))

    # ドット忘れ（別LD結合）
    if '.' in tld:
        typo_variants.append((f"{local}.{tld.replace('.', '')}", "別LD結合（ドット忘れ）"))

    # 別TLD結合（例: co → co.jp）
    tld_alternatives = {
        'com': ['co.jp', 'net'],
        'co.jp': ['com', 'co', 'jp'],
        'co': ['co.jp', 'com'],
    }
    if tld in tld_alternatives:
        for alt in tld_alternatives[tld]:
            typo_variants.append((f"{local}.{alt}", "別TLD結合"))

    # 重複除去
    seen = set()
    unique_typos = []
    for typo, reason in typo_variants:
        if typo not in seen and typo != domain:
            seen.add(typo)
            unique_typos.append((typo, reason))

    return unique_typos



# caese, correctの付与csvファイル出力関数ver1
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

# causeを付け、"domaintypos_dl4_causes.csv"csvファイルに出力
append_typo_causes("filtered_domain_typos_dl4.csv", "domaintypos_dl4_causes.csv")


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


==============================================================================
■ 4. 原因別 差分パターン上位レポート:


【原因: 隣接キー誤打】
  . → ,         : 122件
  p → o         : 62件
  . → /         : 38件
  n → m         : 28件
  o → p         : 24件

【原因: 入力漏れ】
  i → （空）       : 42件
  s → （空）       : 41件
  r → （空）       : 39件
  h → （空）       : 36件
  . → （空）       : 36件

【原因: 別TLD結合】
  p → o         : 54件
  . → （空）       : 15件
  m → .jp       : 13件
  （空） → o         : 6件
  .jp → m         : 6件

【原因: スペルミス（認知ミス）】
  c → s         : 38件
  o → a         : 34件
  e → a         : 25件
  - → ^         : 25件
  a → e         : 22件

【原因: 二重入力】
  （空） → u         : 63件
  （空） → a         : 58件
  （空） → o         : 54件
  （空） → e         : 43件
  （空） → h         : 41件

【原因: ホモグリフ】
  l → i         : 50件
  i → l         : 3件
  ll → ii        : 2件
  . → ,         : 2件
  （空） → o         : 2件

【原因: 視覚類似文字】
  l → i         : 50件
  i → l         : 3件
  ll → ii        : 2件
  . → ,         : 2件
  （空） → o         : 2件

【原因: 入力順序ミス】
  （空） → e         : 8件
  e → （空）       : 8件
  （空） → a         : 7件
  a → （空）       : 7件
  （空） → r         : 4件

【原因: 別LD結合】
  . → ,         : 37件
  . → /         : 25件
  . → -         : 3件
  l → i         : 2件
  . → （空）       : 2件

【原因: ドット抜け】
  . → ,         : 37件
  . → /         : 25件
  . → -         : 3件
  l → i         : 2件
  . → （空）       : 2件

【原因: 左右対称キー誤打】
  d → （空）       : 1件
  （空） → i         : 1件
  d → k         : 1件
  k → d         : 1件

【原因: 別LD結合（ドット忘れ）】
  y → ies       : 1件
  . → （空）       : 1件

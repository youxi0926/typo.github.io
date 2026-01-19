# 先輩にもらったやつ

import pandas as pd
from pyxdameraulevenshtein import damerau_levenshtein_distance
from difflib import SequenceMatcher

def extract_domain(email):
    return email.split('@')[1] if '@' in email else ''

def get_mismatched_part(correct, input_):
    matcher = SequenceMatcher(None, correct, input_)
    mismatched = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            mismatched.append(input_[j1:j2])
    return ''.join(mismatched)

def filter_domain_differences_with_mismatch(input_path, output_path, threshold=5):
    # CSV読み込み
    df = pd.read_csv(input_path)

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

# 使用例
if __name__ == "__main__":
    filter_domain_differences_with_mismatch(
        input_path="filtered_address.csv",
        output_path="filtered_domain_typos_dl7.csv",
        threshold=7
    )

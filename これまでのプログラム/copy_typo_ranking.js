/**
 * @fileoverview Pythonのドメインタイポ予測ロジックをJavaScriptに移植
 */

class TypoRanker {
    constructor(data) {
        // Pythonからエクスポートされた定数と重みデータを格納
        this.keyboardAdjacent = data.keyboard_adjacent;
        this.symmetricKeyPairs = data.symmetric_key_pairs.map(pair => [pair[0], pair[1]]);
        this.homoglyphPairs = data.homoglyph_pairs.map(pair => [pair[0], pair[1]]);
        this.homoglyphsForGenerator = data.homoglyphs_for_generator;
        this.individualWeights = data.individual_weights; // JSON文字列キー形式

        // Damerau-Levenshtein距離の計算ライブラリ（別途導入が必要）
        // ブラウザでは外部ライブラリ (例: 'damerau-levenshtein' npmパッケージ) を利用するか、
        // 独自のDL距離関数を実装する必要があります。
        // ここでは便宜上、シンプルなLevenshtein距離（と転置チェック）を代用します。
        // NOTE: より正確なDL距離が必要な場合は、適切なライブラリを組み込んでください。
        this.damerauLevenshteinDistance = this._levenshteinDistance; 
        this.levenshteinDistance = this._levenshteinDistance;
    }

    /**
     * Levenshtein距離を計算するシンプルな実装 (DL距離の代用)
     * @param {string} a 
     * @param {string} b 
     * @returns {number}
     */
    _levenshteinDistance(a, b) {
        if (a.length === 0) return b.length;
        if (b.length === 0) return a.length;
        const matrix = [];

        for (let i = 0; i <= b.length; i++) {
            matrix[i] = [i];
        }

        for (let j = 0; j <= a.length; j++) {
            matrix[0][j] = j;
        }

        for (let i = 1; i <= b.length; i++) {
            for (let j = 1; j <= a.length; j++) {
                const cost = (a[j - 1] === b[i - 1]) ? 0 : 1;
                matrix[i][j] = Math.min(
                    matrix[i - 1][j] + 1, // 削除
                    matrix[i][j - 1] + 1, // 挿入
                    matrix[i - 1][j - 1] + cost // 置換/一致
                );
            }
        }

        return matrix[b.length][a.length];
    }

    // ======================================================================
    // 汎用ヘルパー関数 (Pythonからの移植)
    // ======================================================================

    keyboardAdjacentCheck(c1, c2) {
        const c1l = c1.toLowerCase();
        return c1l in this.keyboardAdjacent && this.keyboardAdjacent[c1l].includes(c2.toLowerCase());
    }

    isSymmetricMismatch(c1, c2) {
        return this.symmetricKeyPairs.some(([a, b]) => (c1 === a && c2 === b) || (c1 === b && c2 === a));
    }

    isVisualHomoglyph(c1, c2) {
        return this.homoglyphPairs.some(([a, b]) => (c1 === a && c2 === b) || (c1 === b && c2 === a));
    }

    /**
     * 単一の置換・挿入・削除を識別する (SequenceMatcherの簡易的な代替)
     * @param {string} correct 
     * @param {string} typo 
     * @returns {[string, string]} - [correct part, typo part] (例: ['p', 'o'] or ['a', '（空）'])
     */
    identifySingleReplacement(correct, typo) {
        const dl_dist = this.damerauLevenshteinDistance(correct, typo);
        const l_dist = this.levenshteinDistance(correct, typo);

        if (dl_dist > 1 || dl_dist === 0) return ['', '']; // 単一ミスではない or 転置ミスは別途処理

        if (correct.length === typo.length) { // 置換の可能性
            let diffs = [];
            for (let i = 0; i < correct.length; i++) {
                if (correct[i] !== typo[i]) {
                    diffs.push([correct[i], typo[i]]);
                }
            }
            if (diffs.length === 1) return [diffs[0][0], diffs[0][1]];
        } else if (correct.length === typo.length + 1) { // 削除の可能性
            for (let i = 0; i < correct.length; i++) {
                const tempCorrect = correct.slice(0, i) + correct.slice(i + 1);
                if (tempCorrect === typo) return [correct[i], '（空）'];
            }
        } else if (correct.length === typo.length - 1) { // 挿入の可能性
            for (let i = 0; i < typo.length; i++) {
                const tempTypo = typo.slice(0, i) + typo.slice(i + 1);
                if (tempTypo === correct) return ['（空）', typo[i]];
            }
        }

        return ['', ''];
    }

    /**
     * 転置ミスを識別する
     * @param {string} correct 
     * @param {string} typo 
     * @returns {[string, string] | null} - [c1, c2] (例: ['p', 'o'])
     */
    getTransposedPair(correct, typo) {
        // Pythonのロジック: damerau_levenshtein_distance(correct, typo) == 1 and Levenshtein.distance(correct, typo) > 1
        // Damerau-Levenshtein距離の正確な実装がないため、ここでは文字列の単純比較で転置を検出します。
        if (correct.length !== typo.length) return null;

        for (let i = 0; i < correct.length - 1; i++) {
            const tempCorrect = correct.slice(0, i) + correct[i + 1] + correct[i] + correct.slice(i + 2);
            if (tempCorrect === typo) {
                // 正しい文字 c1 c2 が c2 c1 になったと判断
                return [correct[i], correct[i + 1]];
            }
        }
        return null;
    }
    
    // ======================================================================
    // コア機能: タイポ生成とランキング
    // ======================================================================

    /**
     * 個別ミス件数/全タイポ件数 (W_individual) でスコアリングし、ランキングする。
     * @param {string} domain - 正しいドメイン名
     * @param {number} topN - 返すランキングの上位N件
     * @returns {Array<Object>}
     */
    typoGeneratorRanked(domain, topN = 10) {
        const variants = new Map(); // Map<typoString, {causes: Set<string>, score: number}>
        const domainLower = domain.toLowerCase();

        // --- A. タイポ候補の生成 ---
        for (let i = 0; i < domain.length; i++) {
            const c = domain[i];
            const cLower = domainLower[i];
            
            // ヘルパー関数: variantsに追加
            const addVariant = (typo, cause) => {
                if (!variants.has(typo)) {
                    variants.set(typo, { causes: new Set(), score: 0 });
                }
                variants.get(typo).causes.add(cause);
            };

            // 1. 入力漏れ (Deletion)
            const delTy = domain.slice(0, i) + domain.slice(i + 1);
            addVariant(delTy, c === '.' ? "ドット抜け" : "入力漏れ");

            // 2. 二重入力 (Insertion/Repetition)
            const dupTy = domain.slice(0, i) + c + c + domain.slice(i + 1);
            addVariant(dupTy, "二重入力");

            // 3. 隣接キー誤打 (Substitution)
            if (cLower in this.keyboardAdjacent) {
                for (const adj of this.keyboardAdjacent[cLower]) {
                    const adjTy = domain.slice(0, i) + adj + domain.slice(i + 1);
                    addVariant(adjTy, "隣接キー誤打");
                }
            }

            // 4. ホモグリフ・視覚類似文字 (Substitution)
            if (cLower in this.homoglyphsForGenerator) {
                for (const g of this.homoglyphsForGenerator[cLower]) {
                    const hgTy = domain.slice(0, i) + g + domain.slice(i + 1);
                    addVariant(hgTy, "ホモグリフ（視覚類似文字）");
                }
            }
            
            // 5. 左右対称キー誤打 (Substitution)
            for (const [a, b] of this.symmetricKeyPairs) {
                if (c === a) addVariant(domain.slice(0, i) + b + domain.slice(i + 1), "左右対称キー誤打");
                else if (c === b) addVariant(domain.slice(0, i) + a + domain.slice(i + 1), "左右対称キー誤打");
            }

            // 6. 入力順序ミス (Transposition)
            if (i < domain.length - 1) {
                const swapped = domain.slice(0, i) + domain[i + 1] + domain[i] + domain.slice(i + 2);
                addVariant(swapped, "入力順序ミス");
            }
        }

        // --- B. スコアリングとランキング ---
        const rankedResults = [];

        for (const [typo, { causes }] of variants.entries()) {
            if (typo === domain) continue;
            
            let finalScore = 0;
            const distance = this.damerauLevenshteinDistance(domain, typo);
            if (distance > 4) continue; // 距離4超は無視

            const [c1, c2] = this.identifySingleReplacement(domain, typo);

            // 1. 個別ミス重み (W_individual) の適用
            for (const cause of causes) {
                let W_individual = 0.0;
                let key = null;
                const weights = this.individualWeights[cause];
                if (!weights) continue;
                
                // 内部重み (W_individual) の参照ロジック (Pythonの移植)
                if (cause === "入力順序ミス") {
                    const transposedPair = this.getTransposedPair(domain, typo);
                    if (transposedPair) {
                        const [k1, k2] = transposedPair;
                        key = `${k1} ${k2} -> ${k2} ${k1}`; // 文字列キー (ex: 'a b -> b a')
                    }
                } else if (c1 || c2) { // 置換、挿入、削除の単一ミス (c1, c2 の少なくとも一方が非空)
                    // JSONフレンドリーな文字列キー ('p', 'o') -> "po"
                    key = `${c1}${c2}`;
                    if (key.includes('（空）') && key.length > 3) {
                         // 例: '（空）p' または 'p（空）' になるように調整
                         key = c1 === '（空）' ? '（空）' + c2 : c1 + '（空）';
                    } else if (key.includes('（空）') && key.length === 2) {
                        // 例: 'a' + '（空）' -> 'a（空）' になるが、本来は 'a' のみ
                        // Pythonでは ['a', '（空）'] と ['（空）', 'b'] の2つに分かれていたため、
                        // JSON変換後のキーロジックに合わせる必要がある
                        if (c1 === '（空）') key = '（空）' + c2;
                        else key = c1 + '（空）';
                    }
                }

                if (key) {
                    W_individual = weights[key] || 0.0;
                    
                    // 逆順チェック (置換系のみ)
                    if (W_individual === 0.0 && key.length === 2 && !key.includes('（空）')) {
                        W_individual = weights[key.split('').reverse().join('')] || 0.0;
                    }
                }
                
                finalScore += W_individual;
            }

            // 複合ミス ペナルティ
            if (causes.size > 1) finalScore *= 0.5;

            // ボーナス (極小に維持)
            if (causes.size > 0) finalScore += 0.000001;
            
            rankedResults.push({
                typo: typo,
                causes: [...causes].sort().join('・'),
                score: parseFloat(finalScore.toFixed(6)), // 精度を維持
                distance: distance
            });
        }

        // スコア降順、距離昇順でソート
        rankedResults.sort((a, b) => {
            if (b.score !== a.score) {
                return b.score - a.score;
            }
            return a.distance - b.distance;
        });

        return rankedResults.slice(0, topN);
    }
}

// ブラウザでの使用を想定し、グローバルスコープにTypoRankerを公開（またはモジュールとしてエクスポート）
// module.exports = TypoRanker;
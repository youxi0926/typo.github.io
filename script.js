document.addEventListener('DOMContentLoaded', () => {
    fetch('web_data.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('JSONファイルの読み込みに失敗しました: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            // 予測ランキングの表示
            displayPredictionRanking(data.predicted_typos, data.demo_domain);
            
            // 原因別集計の表示
            displayMajorWeights(data.major_weights);
        })
        .catch(error => {
            console.error('データの取得または処理中にエラーが発生しました:', error);
            document.getElementById('ranking-table').querySelector('tbody').innerHTML = 
                `<tr><td colspan="5" style="color: red;">データ表示エラー: ${error.message}</td></tr>`;
        });
});

/**
 * 予測ランキングテーブルを作成する
 * @param {Array<Object>} typos - typo_generator_rankedの出力
 * @param {string} domain - 分析対象のドメイン
 */
function displayPredictionRanking(typos, domain) {
    document.getElementById('demo-domain-name').textContent = domain;
    const tbody = document.getElementById('ranking-table').querySelector('tbody');
    
    typos.forEach((item, index) => {
        const row = tbody.insertRow();
        row.insertCell().textContent = index + 1;
        row.insertCell().textContent = item.typo;
        row.insertCell().textContent = item.score.toFixed(5);
        row.insertCell().textContent = item.distance;
        row.insertCell().textContent = item.causes;
    });
}

/**
 * 主要な原因別割合テーブルを作成する
 * @param {Object} weights - major_weightsのオブジェクト
 */
function displayMajorWeights(weights) {
    const tbody = document.getElementById('major-weights-table').querySelector('tbody');
    
    // スコア降順でソート
    const sortedCauses = Object.entries(weights).sort(([, a], [, b]) => b - a);

    sortedCauses.forEach(([cause, ratio]) => {
        const row = tbody.insertRow();
        row.insertCell().textContent = cause;
        row.insertCell().textContent = ratio.toFixed(3);
    });
}
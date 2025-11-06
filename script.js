document.addEventListener('DOMContentLoaded', () => {
    // web_data.json ファイルを非同期で読み込む
    fetch('web_data.json')
        .then(response => {
            if (!response.ok) {
                // ファイルが見つからない、またはサーバーエラーの場合
                throw new Error('web_data.json の読み込みに失敗しました (' + response.status + ')。ファイルがGitHub Pagesにアップロードされているか確認してください。');
            }
            return response.json();
        })
        .then(data => {
            // データの表示
            displayPredictionRanking(data.predicted_typos, data.demo_domain);
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
 */
function displayPredictionRanking(typos, domain) {
    document.getElementById('demo-domain-name').textContent = domain;
    const tbody = document.getElementById('ranking-table').querySelector('tbody');
    tbody.innerHTML = ''; // 既存の「ロード中です...」をクリア
    
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
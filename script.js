//HTMLファイルの要素を取得するのは、「ロード」後に行なう必要があります。
window.onload = () => {
    document.getElementById("midashi").innerText = "見出し";
}
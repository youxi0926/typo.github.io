from flask import Flask, request, jsonify
from typo_generator import generate_typos  # 既存のロジックを関数化してインポート

app = Flask(__name__)

@app.route("/generate_typos", methods=["POST"])
def generate_typos_api():
    data = request.json
    email = data.get("email", "")
    typos = generate_typos(email)  # あなたの Typo Generator をここで呼ぶ
    return jsonify({"input": email, "typos": typos})

if __name__ == "__main__":
    app.run(debug=True)

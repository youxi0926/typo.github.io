from flask import Flask, request, jsonify
from typo_generator import generate_typos  # あなたの既存のTypo Generator関数を想定

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    email = data.get("email", "")
    typos = generate_typos(email)
    return jsonify({"typos": typos})

if __name__ == "__main__":
    app.run(debug=True)

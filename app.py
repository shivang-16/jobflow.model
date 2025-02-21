from flask import Flask, request, jsonify
from model_manager import load_model

app = Flask(__name__)

@app.route("/<model_version>/predict", methods=["POST"])
def predict(model_version):
    """Handles requests dynamically based on model version"""
    model_module = load_model(model_version)
    if not model_module:
        return jsonify({"error": f"Model {model_version} not found"}), 404

    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400

    response = model_module.generate_response(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

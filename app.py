from flask import Flask, request, jsonify
from model_manager import load_model

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Model Server is running"
    })

@app.route("/predict/<model_version>", methods=["POST"])
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
    print(response, "response in app.py")
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from model_manager import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

    def generate():
        yield "data: {\"status\": \"started\"}\n\n"  # Initial message
        try:
            for token in model_module.generate_response(question):
                if token:  # Only yield non-empty tokens
                    yield f"data: {token}\n\n"
            yield "data: {\"status\": \"completed\"}\n\n"  # Completion message
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return Response(
        stream_with_context(generate()), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Disable buffering in nginx
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

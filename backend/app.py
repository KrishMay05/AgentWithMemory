from flask import Flask, request, jsonify
from flask_cors import CORS
from langgraph_agent import Agent

app = Flask(__name__)
CORS(app)

agent = Agent()

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["prompt"]
    search = request.json["search"]
    response = agent.handle(user_input, search)
    return jsonify(response)

@app.route("/history", methods=["GET"])
def history():
    return jsonify(agent.get_history())

if __name__ == "__main__":
    app.run(port=5050, debug=True)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from llama_cpp import Llama
import markdown

app = Flask(__name__)
CORS(app)

llm = Llama(
    model_path="models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
    chat_format="llama-3",
    n_ctx=1024,
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"},
            {"role": "user", "content": user_message}
        ],
        max_tokens=1024
    )
    assistant_message = response["choices"][0]["message"]["content"]
    assistant_message_html = markdown.markdown(assistant_message)
    return jsonify({"assistant_message": assistant_message_html})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

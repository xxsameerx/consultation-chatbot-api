from flask import Flask, request, jsonify
from chatbot_utils import chunk_data, build_index, load_index, retrieve, generate
import os, json, torch, faiss
from sentence_transformers import SentenceTransformer
from flask_cors import CORS



app = Flask(__name__)
CORS(app)
DATA_FILE = 'comments.json'
IDX_FILE = 'faiss.idx'
PKL_FILE = 'chunks.pkl'
@app.route('/')
def home():
    return "Consultation Chatbot API is running! Use /upload and /ask endpoints."
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    data = json.load(file)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    chunks = chunk_data(data)
    build_index(chunks, IDX_FILE, PKL_FILE)
    return jsonify({'success': True, 'count': len(data)})

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '')
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    index, chunks = load_index(IDX_FILE, PKL_FILE)
    context = retrieve(question, index, embedder, chunks)
    prompt = (
        "You are an assistant that answers ONLY from the consultation comments.\n\n"
        "Comments:\n" + "\n".join(f"- {c}" for c in context) +
        f"\n\nQuestion: {question}\nAnswer:"
    )
    answer = generate(prompt)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

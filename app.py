from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
from fastembed import TextEmbedding
from groq import Groq
import fitz  # pymupdf
import json
import re
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "./pdf_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== Lazy-load: โหลดโมเดล/ฐานข้อมูลตอนมีการเรียกใช้จริงเท่านั้น ====
_model = None
_collection = None


def get_model():
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="intfloat/multilingual-e5-small")
    return _model


def get_collection():
    global _collection
    if _collection is None:
        client_db = chromadb.PersistentClient(path="./database")
        _collection = client_db.get_or_create_collection("physics_exams")
    return _collection


def embed_text(text: str):
    """คืนค่า embedding vector (list ของ float) สำหรับข้อความเดียว"""
    return list(get_model().embed([text]))[0].tolist()


# ดึง API key จาก environment variable แทนการเขียนตรงๆ
client_ai = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def index_pdf(pdf_path: str):
    doc    = fitz.open(pdf_path)
    fname  = os.path.basename(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            chunks.append({"text": text, "page": page_num + 1, "file": fname})
    for chunk in chunks:
        embedding = embed_text(chunk["text"])
        doc_id    = f"{chunk['file']}_page{chunk['page']}"
        get_collection().upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"file": chunk["file"], "page": chunk["page"]}],
        )
    return len(chunks)


def generate_questions(topic: str, level: str, amount: int = 5):
    query_embedding = embed_text(topic)
    results  = get_collection().query(query_embeddings=[query_embedding], n_results=5)
    context  = "\n\n".join(results["documents"][0]) if results["documents"][0] else ""
    level_detail = {
        "1": ("ง่าย",     "ใช้สูตรตรงๆ ค่าตัวเลขง่าย"),
        "2": ("ปานกลาง", "ต้องคิด 2-3 ขั้นตอน"),
        "3": ("ยาก",     "ประยุกต์หลายแนวคิด"),
    }
    level_name, level_desc = level_detail.get(level, ("ปานกลาง", "ต้องคิด 2-3 ขั้นตอน"))
    context_section = f"จากเนื้อหาข้อสอบฟิสิกส์นี้:\n{context}\n\n" if context else ""
    prompt = f"""{context_section}สร้างแบบฝึกหัดฟิสิกส์ระดับ{level_name} หัวข้อ \"{topic}\" จำนวน {amount} ข้อ
เงื่อนไข: {level_desc}
แต่ละข้อต้องมีตัวเลือก A B C D พร้อมระบุข้อที่ถูก

ตอบในรูปแบบ JSON เท่านั้น:
{{
  "questions": [
    {{
      "id": 1,
      "question": "โจทย์ข้อที่ 1",
      "choices": [
        {{"letter": "A", "text": "ตัวเลือก A"}},
        {{"letter": "B", "text": "ตัวเลือก B"}},
        {{"letter": "C", "text": "ตัวเลือก C"}},
        {{"letter": "D", "text": "ตัวเลือก D"}}
      ],
      "correct": "A",
      "answer": "เฉลยและวิธีทำละเอียด"
    }}
  ]
}}"""
    response = client_ai.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = response.choices[0].message.content
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


def check_answer(question: str, correct_answer: str, user_answer: str):
    prompt = f"""โจทย์: {question}
เฉลยที่ถูกต้อง: {correct_answer}
คำตอบของนักเรียน: {user_answer}

ตอบในรูปแบบ JSON เท่านั้น:
{{
  "correct": true หรือ false,
  "score": คะแนน 0-100,
  "feedback": "คำอธิบายสั้นๆ",
  "correct_answer": "เฉลยที่ถูกต้องพร้อมวิธีทำ"
}}"""
    response = client_ai.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์"}), 400
    files   = request.files.getlist("file")
    results = []
    for f in files:
        if not f.filename.endswith(".pdf"):
            results.append({"file": f.filename, "error": "ไม่ใช่ไฟล์ PDF"})
            continue
        save_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(save_path)
        pages = index_pdf(save_path)
        results.append({"file": f.filename, "pages": pages, "success": True})
    return jsonify({"results": results})


@app.route("/list-pdfs", methods=["GET"])
def list_pdfs():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]
    return jsonify({"files": files})


@app.route("/generate", methods=["POST"])
def generate():
    data   = request.json
    result = generate_questions(
        data["topic"],
        data.get("level", "2"),
        int(data.get("amount", 5)),
    )
    return jsonify(result)


@app.route("/check", methods=["POST"])
def check():
    data   = request.json
    result = check_answer(data["question"], data["answer"], data["user_answer"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

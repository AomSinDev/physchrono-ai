from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import fitz  # pymupdf — ใช้แค่อ่านข้อความจาก PDF เบามาก ไม่ต้องพึ่ง torch
import json
import re
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "./pdf_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client_ai = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# เก็บข้อความที่อ่านได้จาก PDF ไว้ในหน่วยความจำ (ไม่ใช้ vector DB)
# โครงสร้าง: { "ชื่อไฟล์.pdf": "ข้อความทั้งหมดในไฟล์" }
_pdf_texts: dict[str, str] = {}


def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc if page.get_text().strip()]
    return "\n\n".join(pages)


def build_context(max_chars: int = 6000) -> str:
    """รวมข้อความจาก PDF ทั้งหมดที่เคยอัปโหลด (ตัดความยาวกันพรอมต์บวมเกินไป)"""
    if not _pdf_texts:
        return ""
    combined = "\n\n".join(_pdf_texts.values())
    return combined[:max_chars]


def generate_questions(topic: str, level: str, amount: int = 5):
    level_detail = {
        "1": ("ง่าย",     "ใช้สูตรตรงๆ ค่าตัวเลขง่าย"),
        "2": ("ปานกลาง", "ต้องคิด 2-3 ขั้นตอน"),
        "3": ("ยาก",     "ประยุกต์หลายแนวคิด"),
    }
    level_name, level_desc = level_detail.get(level, ("ปานกลาง", "ต้องคิด 2-3 ขั้นตอน"))

    context = build_context()
    context_section = f"อ้างอิงจากเนื้อหาข้อสอบฟิสิกส์นี้:\n{context}\n\n" if context else ""

    prompt = f"""{context_section}สร้างแบบฝึกหัดฟิสิกส์ระดับ{level_name} หัวข้อ "{topic}" จำนวน {amount} ข้อ
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
        text = extract_pdf_text(save_path)
        _pdf_texts[f.filename] = text
        results.append({"file": f.filename, "chars": len(text), "success": True})
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

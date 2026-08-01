from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import fitz  # pymupdf — อ่านข้อความจาก PDF เบามาก ไม่ต้องพึ่ง torch
import json
import re
import os
import tempfile

app = Flask(__name__)
CORS(app)

client_ai = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MAX_CONTEXT_CHARS = 8000  # กันพรอมต์ยาวเกินไป


def extract_pdf_text(file_storage) -> str:
    """อ่านข้อความจากไฟล์ PDF ที่อัปโหลดมา (ไม่บันทึกถาวร ใช้แค่ตอนคำขอนี้)"""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        file_storage.save(tmp.name)
        doc = fitz.open(tmp.name)
        pages = [page.get_text() for page in doc if page.get_text().strip()]
        return "\n\n".join(pages)


def generate_questions(topic: str, level: str, amount: int = 5, context: str = ""):
    level_detail = {
        "1": ("ง่าย",     "ใช้สูตรตรงๆ ค่าตัวเลขง่าย"),
        "2": ("ปานกลาง", "ต้องคิด 2-3 ขั้นตอน"),
        "3": ("ยาก",     "ประยุกต์หลายแนวคิด"),
    }
    level_name, level_desc = level_detail.get(level, ("ปานกลาง", "ต้องคิด 2-3 ขั้นตอน"))

    context_section = ""
    if context:
        context_section = f"อ้างอิงจากเนื้อหาในเอกสารนี้:\n{context[:MAX_CONTEXT_CHARS]}\n\n"

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


@app.route("/extract-pdf", methods=["POST"])
def extract_pdf():
    """รับไฟล์ PDF มาแล้วคืนข้อความล้วนกลับไปทันที ไม่เก็บไว้ที่ server"""
    if "file" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์"}), 400

    files = request.files.getlist("file")
    combined_text = []
    file_names = []

    for f in files:
        if not f.filename.endswith(".pdf"):
            continue
        try:
            text = extract_pdf_text(f)
            combined_text.append(text)
            file_names.append(f.filename)
        except Exception as e:
            return jsonify({"error": f"อ่านไฟล์ {f.filename} ไม่สำเร็จ: {str(e)}"}), 400

    if not combined_text:
        return jsonify({"error": "ไม่พบไฟล์ PDF ที่ใช้ได้"}), 400

    full_text = "\n\n".join(combined_text)
    return jsonify({
        "text": full_text[:MAX_CONTEXT_CHARS],
        "files": file_names,
        "chars": len(full_text),
    })


@app.route("/generate", methods=["POST"])
def generate():
    data   = request.json
    result = generate_questions(
        data["topic"],
        data.get("level", "2"),
        int(data.get("amount", 5)),
        data.get("context", ""),
    )
    return jsonify(result)


@app.route("/check", methods=["POST"])
def check():
    data   = request.json
    result = check_answer(data["question"], data["answer"], data["user_answer"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

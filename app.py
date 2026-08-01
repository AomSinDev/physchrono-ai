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


_SUPERSCRIPT_MAP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def _to_superscript(text: str) -> str:
    """แปลงเลขยกกำลังแบบ ^2 หรือ ^-1 ให้เป็นสัญลักษณ์ยกกำลังจริง (²) แทนเครื่องหมาย caret"""
    def replace(match):
        exponent = match.group(1)
        return exponent.translate(_SUPERSCRIPT_MAP)
    return re.sub(r"\^(-?\d+)", replace, text)


def _fix_superscripts(questions: list) -> list:
    for q in questions:
        if "question" in q:
            q["question"] = _to_superscript(q["question"])
        if "answer" in q:
            q["answer"] = _to_superscript(q["answer"])
        for choice in q.get("choices", []):
            if "text" in choice:
                choice["text"] = _to_superscript(choice["text"])
    return questions


_FOREIGN_CHAR_PATTERN = re.compile(
    r"[\u4e00-\u9fff"   # CJK Unified Ideographs (จีน)
    r"\u3040-\u30ff"    # Hiragana/Katakana (ญี่ปุ่น)
    r"\uac00-\ud7a3"    # Hangul (เกาหลี)
    r"]"
)


def _contains_foreign_chars(questions: list) -> bool:
    for q in questions:
        texts = [q.get("question", ""), q.get("answer", "")]
        texts += [c.get("text", "") for c in q.get("choices", [])]
        for t in texts:
            if _FOREIGN_CHAR_PATTERN.search(t):
                return True
    return False


def _strip_foreign_chars(questions: list) -> list:
    """กันไว้ชั้นสุดท้าย: ถ้า AI ยังสร้างตัวอักษรแปลกปลอมซ้ำหลายรอบ ให้ตัดออกเลยแทนที่จะปล่อยผ่าน"""
    for q in questions:
        if "question" in q:
            q["question"] = _FOREIGN_CHAR_PATTERN.sub("", q["question"])
        if "answer" in q:
            q["answer"] = _FOREIGN_CHAR_PATTERN.sub("", q["answer"])
        for choice in q.get("choices", []):
            if "text" in choice:
                choice["text"] = _FOREIGN_CHAR_PATTERN.sub("", choice["text"])
    return questions


def _fix_correct_letters(questions: list) -> list:
    """
    AI บางครั้งคำนวณเลขถูกในคำอธิบาย แต่แปะป้าย 'correct' ผิดตัวเลือก
    ฟังก์ชันนี้เทียบค่าตัวเลขจริง (correct_value) กับตัวเลขในแต่ละตัวเลือก
    แล้วแก้ป้าย correct ให้ตรงกับตัวเลือกที่ใกล้เคียงค่าจริงที่สุด
    """
    num_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")

    for q in questions:
        correct_value = q.pop("correct_value", None)
        if correct_value is None:
            continue
        try:
            target = float(correct_value)
        except (TypeError, ValueError):
            continue

        best_letter = None
        best_diff = None
        for choice in q.get("choices", []):
            match = num_pattern.search(str(choice.get("text", "")))
            if not match:
                continue
            try:
                value = float(match.group())
            except ValueError:
                continue
            diff = abs(value - target)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_letter = choice.get("letter")

        if best_letter:
            q["correct"] = best_letter

    return questions


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

กฎสำคัญที่ต้องทำตามทุกข้อ:
1. คำนวณคำตอบที่ถูกต้องให้แม่นยำก่อน แล้วค่อยใส่เป็นหนึ่งในตัวเลือก A-D (ห้ามให้ตัวเลือกที่ถูกคลาดเคลื่อนจากค่าที่คำนวณได้จริง)
2. ฟิลด์ "correct" ต้องตรงกับตัวอักษรของตัวเลือกที่มีค่าตรงกับคำตอบที่คำนวณได้จริงเท่านั้น ห้ามเดาหรือเลือกแบบประมาณ
3. ใส่ฟิลด์ "correct_value" เป็นตัวเลขล้วน (ไม่มีหน่วย ไม่มีข้อความ) ของคำตอบที่ถูกต้องจริงๆ ที่คำนวณได้ เพื่อใช้ตรวจสอบ
4. เขียนคำอธิบายในฟิลด์ "answer" แบบมั่นใจ ตรงไปตรงมา แสดงวิธีคำนวณทีละขั้นตอน ห้ามเขียนลังเลหรือขัดแย้งกันเอง (เช่น ห้ามเขียนทำนอง "แต่ตัวเลือกที่ใกล้ที่สุดคือ...")
5. เขียนโจทย์และคำอธิบายเป็นภาษาไทยล้วน ห้ามใส่คำภาษาอังกฤษปนที่ไม่มีความหมาย (เช่น คำย่อแปลกๆ) ยกเว้นสัญลักษณ์หน่วยสากลมาตรฐานเท่านั้น เช่น m, s, kg, N, J, m/s, m/s^2
6. หน่วยที่มีเลขยกกำลัง ให้เขียนด้วยสัญลักษณ์หน่วยสากลแบบย่อเสมอ เช่น "m/s^2" (จะถูกแปลงเป็น m/s² อัตโนมัติ) ห้ามสะกดหน่วยเป็นคำไทยยาวๆ แบบ "เมตร/วินาที^2"
7. ห้ามใช้ตัวอักษรจีน ญี่ปุ่น เกาหลี หรือภาษาอื่นใดนอกจากไทยและอังกฤษเด็ดขาด ไม่ว่ากรณีใดก็ตาม

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
      "correct_value": 0,
      "answer": "เฉลยและวิธีทำละเอียด"
    }}
  ]
}}"""
    max_attempts = 3
    result = None
    for attempt in range(max_attempts):
        response = client_ai.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = response.choices[0].message.content
        text = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)

        if not _contains_foreign_chars(result.get("questions", [])):
            break
        # ถ้าเจอตัวอักษรแปลกปลอม ลองสร้างใหม่อีกรอบ (ยกเว้นรอบสุดท้าย)

    result["questions"] = _fix_correct_letters(result.get("questions", []))
    result["questions"] = _fix_superscripts(result["questions"])
    if _contains_foreign_chars(result["questions"]):
        # สร้างใหม่ครบ 3 รอบแล้วยังไม่สะอาด — ตัดตัวอักษรแปลกปลอมทิ้งเป็นทางสุดท้าย
        result["questions"] = _strip_foreign_chars(result["questions"])
    return result


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

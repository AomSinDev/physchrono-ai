import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# โหลดโมเดลและฐานข้อมูล
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client_db = chromadb.PersistentClient(path="./database")
collection = client_db.get_or_create_collection("physics_exams")

# Groq API Key
client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_questions(topic, level, amount=5):
    query_embedding = model.encode(topic).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    context = "\n\n".join(results["documents"][0])
    
    level_detail = {
        "1": ("ง่าย", "ใช้สูตรตรงๆ ค่าตัวเลขง่าย ไม่ต้องแปลงหน่วย"),
        "2": ("ปานกลาง", "ต้องคิด 2-3 ขั้นตอน มีการแปลงหน่วย"),
        "3": ("ยาก", "ประยุกต์หลายแนวคิด คิดหลายขั้นตอน โจทย์ซับซ้อน")
    }
    
    level_name, level_desc = level_detail[level]
    
    prompt = f"""จากเนื้อหาข้อสอบฟิสิกส์นี้:
{context}

สร้างแบบฝึกหัดระดับ{level_name}หัวข้อ "{topic}" จำนวน {amount} ข้อ
เงื่อนไข: {level_desc}
แสดงเฉลยทุกข้อด้วย"""

    response = client_ai.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    print("=== ระบบออกโจทย์ฟิสิกส์ Phy-Chrono ===")
    topic = input("ป้อนหัวข้อ: ")
    
    print("\nเลือกระดับความยาก:")
    print("1 = ง่าย")
    print("2 = ปานกลาง")
    print("3 = ยาก")
    level = input("เลือก (1/2/3): ")
    
    amount = input("ต้องการกี่ข้อ (กด Enter = 5 ข้อ): ")
    amount = int(amount) if amount else 5
    
    print("\nกำลังสร้างโจทย์...\n")
    result = generate_questions(topic, level, amount)
    print(result)
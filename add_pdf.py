import fitz  # pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
import os
import sys

# โหลดโมเดลแปลงข้อความ
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# เชื่อมต่อฐานข้อมูล
client = chromadb.PersistentClient(path="./database")
collection = client.get_or_create_collection("physics_exams")

def add_pdf(pdf_path):
    print(f"กำลังอ่าน: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            chunks.append({
                "text": text,
                "page": page_num + 1,
                "file": os.path.basename(pdf_path)
            })
    
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk["text"]).tolist()
        doc_id = f"{chunk['file']}_page{chunk['page']}"
        
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"file": chunk["file"], "page": chunk["page"]}]
        )
    
    print(f"✅ เพิ่มสำเร็จ {len(chunks)} หน้า จากไฟล์ {os.path.basename(pdf_path)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        add_pdf(sys.argv[1])
    else:
        for f in os.listdir("./pdf_files"):
            if f.endswith(".pdf"):
                add_pdf(f"./pdf_files/{f}")
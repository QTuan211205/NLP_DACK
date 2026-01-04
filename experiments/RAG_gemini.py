import requests
import os
import json
import time
from dotenv import load_dotenv

# --- IMPORTS ---
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("wordnet")

from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import GraphCypherQAChain
import google.generativeai as genai

rouge = Rouge()

# ==============================================================================
# 1. CẤU HÌNH & KẾT NỐI
# ==============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '..', 'key.env')
load_dotenv(env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URI = os.getenv("URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("USER", "neo4j")
NEO4J_PASSWORD = os.getenv("PASSWORD", "12345678")

if not GOOGLE_API_KEY:
    print("❌ LỖI: Không tìm thấy GOOGLE_API_KEY.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "models/gemini-2.0-flash" # nhớ sửa model lại 

try:
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    graph.refresh_schema()
    print("✅ Đã kết nối Neo4j! (Schema đã khớp với dữ liệu Tiếng Việt)")
except Exception as e:
    print(f"❌ Lỗi kết nối Neo4j: {e}")
    exit()

# ==============================================================================
# 2. PROMPT & SCHEMA (QUAN TRỌNG NHẤT: DÙNG KEY TIẾNG VIỆT)
# ==============================================================================

# Ví dụ mẫu dạy Bot cách query sang bảng TIÊU_CHUẨN
examples = [
    {
        "question": "Công thức hóa học của Aspirin là gì?",
        # SỬA: Trả về cả tên hoạt chất để Bot biết công thức này của ai
        "query": "MATCH (n:HOẠT_CHẤT) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('ASPIRIN') RETURN n.tên_hoạt_chất,n.công_thức_hóa_học",
    },
    {
        "question": "Công thức hóa học của Acid Ascorbic?",
        "query": "MATCH (n:HOẠT_CHẤT) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('ACID ASCORBIC') RETURN n.tên_hoạt_chất,n.công_thức_hóa_học",
    },
    {
        "question": "Mô tả chung về Paracetamol?",
        "query": "MATCH (n:HOẠT_CHẤT) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('PARACETAMOL') RETURN n.tên_hoạt_chất,n.mô_tả",
    },
    {
        "question": "Yêu cầu về định lượng của Bột bó?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('BỘT BÓ') RETURN n.tên_hoạt_chất,t.định_lượng",
    },
    {
        "question": "Độ hòa tan của Glucose?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('GLUCOSE') RETURN n.tên_hoạt_chất,t.độ_hòa_tan",
    },
    {
        "question": "Bột bó thuộc loại thuốc nào?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:THUỘC_NHÓM]->(l:LOẠI_THUỐC) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('BỘT BÓ') RETURN n.tên_hoạt_chất,l.tên_loại",
    }
]

# Khai báo cấu trúc đúng với Database hiện tại của bạn
PREFIX = """
    You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query.
    
    My Database Schema (Tiếng Việt):
    
    1. Node: HOẠT_CHẤT
       - tên_hoạt_chất
       - tên_latin
       - công_thức_hóa_học
       - mô_tả
       - bảo_quản
       
    2. Node: TIÊU_CHUẨN (Linked via :CÓ_TIÊU_CHUẨN)
       - định_lượng
       - định_tính
       - độ_hòa_tan
       - tạp_chất_và_độ_tinh_khiết
       - hàm_lượng_yêu_cầu

    3. Node: LOẠI_THUỐC (Linked via :THUỘC_NHÓM)
       - tên_loại

    INSTRUCTIONS:
    - Use `toLower()` for case-insensitive search.
    - Use `CONTAINS` for fuzzy matching.
    - IMPORTANT: Use the EXACT Vietnamese property names listed above (e.g. `n.tên_hoạt_chất`, `t.định_lượng`).
    - If asked about quantitative standards (định lượng/hòa tan), YOU MUST JOIN with `[:CÓ_TIÊU_CHUẨN]`.
    
    Examples:
"""

example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=PREFIX,
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question"],
)

gemini_chain = GraphCypherQAChain.from_llm(
    ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0),
    graph=graph,
    verbose=True,
    cypher_prompt=prompt,
    allow_dangerous_requests=True
)

# ==============================================================================
# 3. CHẠY THỰC NGHIỆM
# ==============================================================================

results_dir = "../results"
logs_dir = "../logs"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
gemini_results_path = os.path.join(results_dir, "gemini_results.txt")
gemini_log_path = os.path.join(logs_dir, "gemini_log.json")
gemini_log = []

def get_gemini_fallback(text):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        return model.generate_content([text]).text
    except: return "Lỗi kết nối Gemini."

print("\n🚀 BẮT ĐẦU CHẠY THỬ NGHIỆM RAG (ĐÃ FIX SCHEMA TIẾNG VIỆT)...")

# Bộ câu hỏi test
data_test = [
    # 1. Nhóm câu hỏi về ĐỊNH LƯỢNG (Yêu cầu Bot phải tìm trong bảng TIÊU_CHUẨN)
    {
        "question": "Yêu cầu định lượng đối với VIÊN NÉN ACID ACETYLSALICYLIC là gì?", 
        "answer": "Hàm lượng C9H8O4 từ 95,0 % đến 105,0 % so với lượng ghi trên nhãn."
    },
    {
        "question": "Giới hạn định lượng của ACID AMINOCAPROIC được quy định như thế nào?", 
        "answer": "Hàm lượng C6H13NO2 phải đạt từ 99,0 % đến 101,0 %."
    },
    
    # 2. Nhóm câu hỏi về TÍNH CHẤT / MÔ TẢ (Kiểm tra khả năng đọc hiểu văn bản dài)
    {
        "question": "Mô tả tính chất vật lý của ACID CITRIC NGẬM MỘT PHÂN TỬ NƯỚC?", 
        "answer": "Tinh thể không màu hoặc bột kết tinh trắng, sủi bọt trong không khí khô."
    },
    {
        "question": "Đặc điểm cảm quan của BỘT PHA HỖN DỊCH AZITHROMYCIN?", 
        "answer": "Bột khô, tơi, màu trắng hoặc trắng ngà, mùi thơm đặc trưng."
    },
    
    # 3. Nhóm câu hỏi về BẢO QUẢN (Dữ liệu nằm trực tiếp ở node HOẠT_CHẤT)
    {
        "question": "Cách bảo quản thuốc BẠC VITELINAT như thế nào?", 
        "answer": "Đựng trong lọ màu, nút kín, để chỗ tối."
    },
    
    # 4. Nhóm câu hỏi về ĐỊNH TÍNH (Nhận biết hoạt chất)
    {
        "question": "Phản ứng định tính để nhận biết ACID ASCORBIC?", 
        "answer": "Làm mất màu dung dịch 2,6-diclorophenolindophenol hoặc tủa với bạc nitrat."
    },
    
    # 5. Nhóm câu hỏi về PHÂN LOẠI (Mối quan hệ THUỘC_NHÓM)
    {
        "question": "BỘT PHA HỖN DỊCH AMOXICILIN VÀ ACID CLAVULANIC thuộc nhóm thuốc nào?", 
        "answer": "Nhóm kháng sinh beta-lactam."
    }
]

for i, x in enumerate(data_test):
    print(f"\n🔹 Câu hỏi {i+1}: {x['question']}")
    try:
        # Chạy Chain
        response = gemini_chain.invoke(x["question"])
        gemini_result = response.get('result', str(response))
    except Exception as e:
        print(f"   ⚠️ Lỗi Cypher: {e}")
        gemini_result = get_gemini_fallback(f"Dược điển: {x['question']}")
    
    if "I don't know" in str(gemini_result):
        gemini_result = "Không tìm thấy trong DB (Vẫn lỗi khớp tên)."
        
    print(f"✅ Trả lời: {gemini_result}")
    
    # Ghi log đơn giản
    gemini_log.append({
        "question": x["question"],
        "answer": gemini_result,
        "cypher_used": "Xem trong log console"
    })

# Lưu log
with open(gemini_log_path, "w", encoding='utf-8') as f:
    json.dump(gemini_log, f, ensure_ascii=False, indent=4)

print("\n🎉 HOÀN TẤT! Hãy kiểm tra kết quả phía trên.")
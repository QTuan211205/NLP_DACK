import google.generativeai as genai
from py2neo import Graph
import json

# ==========================================
# 1. CẤU HÌNH
# ==========================================
# Hãy dán API Key của bạn vào giữa dấu ngoặc kép bên dưới
GOOGLE_API_KEY = "".strip()

if "DÁN_API_KEY" in GOOGLE_API_KEY:
    print("❌ LỖI: Bạn chưa điền API Key vào file code!")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

# Kết nối Neo4j
try:
    graph = Graph("neo4j://127.0.0.1:7687", auth=("neo4j", "12345678"))
    print("✅ Đã kết nối Neo4j thành công!")
except Exception as e:
    print(f"❌ Lỗi kết nối Neo4j: {e}")
    print("👉 Hãy chắc chắn bạn đã bật Neo4j Desktop (nút Start) chưa?")
    exit()

# Chọn model (Dùng bản 2.5 flash như bạn đã check thành công)
model = genai.GenerativeModel('models/gemini-2.5-flash')

# ==========================================
# 2. ĐỊNH NGHĨA SCHEMA (SỬA LẠI CHUẨN BACKTICK)
# ==========================================
schema_desc = """
Đây là cấu trúc Graph Database y khoa (Neo4j). 
LƯU Ý ĐẶC BIỆT: Tên Nhãn (Label) và Quan hệ (Relationship) ĐỀU CÓ DẤU CÁCH.
Bắt buộc phải dùng dấu huyền (backtick `) bao quanh tên.

Nodes (Nhãn có dấu cách):
- (:`BỆNH`) {tên_bệnh, mô_tả_bệnh, nguyên_nhân, loại_bệnh, cách_phòng_tránh}
- (:`THUỐC`) {tên_bệnh, thuốc_phổ_biến, đề_xuất_thuốc, thông_tin_thuốc}
- (:`TRIỆU CHỨNG`) {tên_bệnh, triệu_chứng, kiểm_tra, đối_tượng_dễ_mắc_bệnh}
- (:`LỜI KHUYÊN`) {tên_bệnh, nên_ăn_thực_phẩm_chứa, không_nên_ăn_thực_phẩm_chứa, đề_xuất_món_ăn}
- (:`ĐIỀU TRỊ`) {tên_bệnh, phương_pháp, khoa_điều_trị, tỉ_lệ_chữa_khỏi}

Relationships (Quan hệ có dấu cách):
- (:`BỆNH`)-[:`CÓ TRIỆU CHỨNG`]->(:`TRIỆU CHỨNG`)
- (:`BỆNH`)-[:`ĐƯỢC KÊ ĐƠN`]->(:`THUỐC`)
- (:`BỆNH`)-[:`ĐIỀU TRỊ VÀ PHÒNG TRÁNH CÙNG`]->(:`LỜI KHUYÊN`)
- (:`BỆNH`)-[:`ĐƯỢC CHỮA BỞI`]->(:`ĐIỀU TRỊ`)
- (:`BỆNH`)-[:`ĐI KÈM VỚI BỆNH`]->(:`BỆNH`)
"""

# ==========================================
# 3. CÁC HÀM XỬ LÝ CHÍNH
# ==========================================
def generate_cypher(question):
    """Bước 1: Chuyển câu hỏi thành Cypher Query"""
    print("   ↳ 🤖 Đang suy nghĩ câu lệnh truy vấn...")
    prompt = f"""
    Bạn là chuyên gia Neo4j. Hãy viết câu lệnh Cypher để trả lời câu hỏi.
    
    Schema: {schema_desc}
    Câu hỏi: "{question}"
    
    QUY TẮC BẮT BUỘC (TUÂN THỦ 100%):
    1. VÌ TÊN CÓ DẤU CÁCH, BẮT BUỘC PHẢI DÙNG DẤU HUYỀN (`) ĐỂ BAO QUANH TÊN NHÃN VÀ QUAN HỆ.
       - ĐÚNG: MATCH (b:`BỆNH`)-[:`CÓ TRIỆU CHỨNG`]->(t:`TRIỆU CHỨNG`)
       - SAI:  MATCH (b:BỆNH)-[:CÓ_TRIỆU_CHỨNG]->(t:TRIỆU CHỨNG)
       - SAI:  MATCH (b:BỆNH)-[:'CÓ TRIỆU CHỨNG']->(t:'TRIỆU CHỨNG')
       
    2. Dùng `CONTAINS` cho tìm kiếm tên bệnh (b.tên_bệnh) để tìm kiếm linh hoạt.
    3. Chỉ trả về code Cypher, không giải thích.
    4. Luôn RETURN các thuộc tính cần thiết để trả lời.
    """
    response = model.generate_content(prompt)
    # Làm sạch response (xóa markdown nếu có)
    query = response.text.strip().replace("```cypher", "").replace("```", "")
    return query

def generate_answer(question, data):
    """Bước 2: Tổng hợp câu trả lời từ dữ liệu"""
    print("   ↳ 👩‍⚕️ Đang tổng hợp câu trả lời...")
    prompt = f"""
    Dữ liệu từ Database y khoa: {json.dumps(data, ensure_ascii=False)}
    Câu hỏi người dùng: "{question}"
    
    Hãy đóng vai Bác sĩ ảo, trả lời người dùng một cách tự nhiên, chi tiết và thân thiện bằng tiếng Việt.
    - Nếu dữ liệu rỗng (empty), hãy nói "Xin lỗi, tôi chưa có thông tin về vấn đề này trong hệ thống."
    - Đừng chỉ liệt kê, hãy viết thành câu văn mạch lạc.
    """
    response = model.generate_content(prompt)
    return response.text

def chat_with_kg(user_question):
    print(f"\n👤 User: {user_question}")
    
    try:
        # B1: Tạo Query
        cypher_query = generate_cypher(user_question)
        # Uncomment dòng dưới nếu muốn xem lệnh Cypher sinh ra
        # print(f"DEBUG Query: {cypher_query}") 
        
        # B2: Chạy Query
        results = graph.run(cypher_query).data()
        print(f"📂 Tìm thấy: {len(results)} bản ghi thông tin.")
        
        # B3: Trả lời
        final_answer = generate_answer(user_question, results)
        print(f"🏥 Assistant: {final_answer}")
        return final_answer
        
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")
        return "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi."

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    print("="*50)
    print("CHÀO MỪNG BẠN ĐẾN VỚI CHATBOT Y KHOA VIETMEDKG")
    print("="*50)
    
    # Chạy thử 1 câu mẫu
    # chat_with_kg("Bệnh Ho gà có triệu chứng gì?")

    while True:
        q = input("\n💬 Mời bạn đặt câu hỏi (hoặc gõ 'exit' để thoát): ")
        if q.lower() in ['exit', 'quit', 'thoát']:
            print("👋 Tạm biệt!")
            break
        if q.strip() == "": continue
        
        chat_with_kg(q)
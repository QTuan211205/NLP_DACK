import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from py2neo import Graph
import google.generativeai as genai
import numpy as np

# ==========================================
# 1. CẤU HÌNH
# ==========================================
GOOGLE_API_KEY = "".strip()
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel('models/gemini-2.5-flash')

# Kết nối Neo4j
try:
    graph = Graph("bolt://127.0.0.1:7687", auth=("neo4j", "12345678"))
    print("✅ Đã kết nối Neo4j!")
except:
    print("❌ Không kết nối được Neo4j. Hãy bật Neo4j Desktop!")
    exit()

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU TÌM KIẾM (INDEXING)
# ==========================================
print("⏳ Đang tải dữ liệu và khởi tạo mô hình tìm kiếm...")

# Load dữ liệu gốc để làm danh sách tìm kiếm (Search Corpus)
# Đường dẫn này trỏ đến file CSV bạn dùng để import Neo4j
df = pd.read_csv(r'..\data\data_translated.csv', encoding='utf-8')

# Chúng ta sẽ tìm kiếm trên cột 'tên_bệnh' (Entity Name)
# Nếu bạn muốn tìm cả thuốc, hãy gộp thêm danh sách thuốc vào đây
corpus = df['tên_bệnh'].dropna().unique().tolist()
corpus = [str(x).strip() for x in corpus if str(x).strip()]

# --- A. Cấu hình Vector Search (Semantic) ---
# Dùng model nhỏ gọn hỗ trợ tiếng Việt tốt
embedder = SentenceTransformer('keepitreal/vietnamese-sbert') 
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# --- B. Cấu hình Keyword Search (BM25) ---
# Tách từ đơn giản bằng khoảng trắng (có thể dùng pyvi nếu muốn chuẩn hơn)
tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

print(f"✅ Đã index xong {len(corpus)} thực thể bệnh.")

# ==========================================
# 3. THUẬT TOÁN HYBRID SEARCH (RRF)
# ==========================================
def hybrid_search(query, top_k=3):
    """
    Kết hợp Vector Search và BM25 bằng thuật toán Reciprocal Rank Fusion (RRF)
    """
    # 1. Vector Search Results
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    # Lấy top 10 vector
    search_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=10)[0]
    
    # 2. BM25 Search Results
    tokenized_query = query.lower().split(" ")
    # Lấy top 10 BM25 (trả về danh sách text, cần map ngược lại index)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_n = np.argsort(bm25_scores)[::-1][:10]
    
    # 3. RRF Fusion (Tính điểm xếp hạng)
    # Công thức: score = 1 / (k + rank)
    rrf_score = {}
    k = 60 # Hằng số thường dùng trong RRF
    
    # Cộng điểm từ Vector
    for rank, hit in enumerate(search_hits):
        doc_idx = hit['corpus_id']
        rrf_score[doc_idx] = rrf_score.get(doc_idx, 0) + (1 / (k + rank + 1))
        
    # Cộng điểm từ BM25
    for rank, doc_idx in enumerate(bm25_top_n):
        rrf_score[doc_idx] = rrf_score.get(doc_idx, 0) + (1 / (k + rank + 1))
        
    # Sắp xếp lại theo điểm RRF cao nhất
    sorted_rrf = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)
    
    # Lấy top_k kết quả cuối cùng
    final_results = []
    for doc_idx, score in sorted_rrf[:top_k]:
        final_results.append(corpus[doc_idx])
        
    return final_results

# ==========================================
# 4. TRUY VẤN GRAPH & SINH CÂU TRẢ LỜI
# ==========================================
def get_graph_context(disease_name):
    """
    Khi đã biết chính xác tên bệnh, truy vấn thẳng vào Neo4j
    """
    query = f"""
    MATCH (b:`BỆNH` {{tên_bệnh: "{disease_name}"}})
    OPTIONAL MATCH (b)-[:`CÓ TRIỆU CHỨNG`]->(tc:`TRIỆU CHỨNG`)
    OPTIONAL MATCH (b)-[:`ĐIỀU TRỊ VÀ PHÒNG TRÁNH CÙNG`]->(lk:`LỜI KHUYÊN`)
    OPTIONAL MATCH (b)-[:`ĐƯỢC KÊ ĐƠN`]->(t:`THUỐC`)
    OPTIONAL MATCH (b)-[:`ĐƯỢC CHỮA BỞI`]->(dt:`ĐIỀU TRỊ`)
    RETURN b, tc, lk, t, dt
    """
    return graph.run(query).data()

def generate_answer(user_question, context_data, disease_found):
    prompt = f"""
    Bạn là bác sĩ AI. Người dùng đang hỏi về: "{user_question}"
    Hệ thống tìm kiếm đã xác định bệnh liên quan nhất là: "{disease_found}"
    
    Dữ liệu chi tiết từ Knowledge Graph:
    {context_data}
    
    Hãy trả lời câu hỏi dựa trên dữ liệu trên. Nếu dữ liệu không đủ, hãy nói rõ.
    """
    response = model_gemini.generate_content(prompt)
    return response.text

# ==========================================
# 5. CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    print("\n🚀 HỆ THỐNG HYBRID RAG ĐÃ SẴN SÀNG!")
    print("Mô hình này kết hợp tìm kiếm Vector + Từ khóa để tìm đúng tên bệnh trước khi tra cứu.")
    
    while True:
        question = input("\n👤 Bạn hỏi: ")
        if question.lower() in ['exit', 'quit']: break
        
        # B1: Hybrid Search để tìm thực thể (Entity Linking)
        top_matches = hybrid_search(question, top_k=1)
        
        if not top_matches:
            print("🤖 Bot: Xin lỗi, tôi không tìm thấy tên bệnh nào khớp trong dữ liệu.")
            continue
            
        best_match = top_matches[0]
        print(f"🔍 Hệ thống xác định chủ đề: '{best_match}'")
        
        # B2: Lấy dữ liệu Graph
        context = get_graph_context(best_match)
        
        # B3: Gemini trả lời
        if context:
            answer = generate_answer(question, context, best_match)
            print(f"🏥 Bot đáp: {answer}")
        else:
            print(f"🤖 Bot: Tìm thấy tên bệnh '{best_match}' nhưng chưa có dữ liệu chi tiết trong Graph.")
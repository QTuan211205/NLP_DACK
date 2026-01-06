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

# Tải dữ liệu wordnet nếu chưa có
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("wordnet")

from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import GraphCypherQAChain
import google.generativeai as genai

# Khởi tạo Rouge
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
MODEL_NAME = "gemini-2.0-flash" 

try:
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    graph.refresh_schema()
    print("✅ Đã kết nối Neo4j!")
except Exception as e:
    print(f"❌ Lỗi kết nối Neo4j: {e}")
    exit()

# ==============================================================================
# 2. PROMPT & SCHEMA
# ==============================================================================

examples = [
    # --- 1-HOP (Căn bản) ---
    {
        "question": "Công thức hóa học của Aspirin là gì?",
        "query": "MATCH (n:HOẠT_CHẤT) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('ASPIRIN') RETURN n.tên_hoạt_chất, n.công_thức_hóa_học",
    },
    # --- 2-HOP (Xé nhỏ từ khóa tính chất) ---
    {
        "question": "Dược chất có tính chất [bột trắng, tan trong nước, không tan trong ethanol] có định tính là gì?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tính_chất) CONTAINS 'bột' AND toLower(n.tính_chất) CONTAINS 'trắng' AND toLower(n.tính_chất) CONTAINS 'nước' AND toLower(n.tính_chất) CONTAINS 'ethanol' RETURN n.tên_hoạt_chất, t.định_tính, t.định_lượng",
    },
    {
        "question": "Quy trình định tính cho dược chất có đặc tính [tnc ~143°C, dễ tan trong nước và ethanol]?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE n.tính_chất CONTAINS '143' AND toLower(n.tính_chất) CONTAINS 'nước' AND toLower(n.tính_chất) CONTAINS 'ethanol' RETURN n.tên_hoạt_chất, t.định_tính, t.độ_hòa_tan",
    },
    {
        "question": "Tìm hoạt chất là [tinh thể không màu, khó tan trong nước] và thuộc loại thuốc gì?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:THUỘC_NHÓM]->(l:LOẠI_THUỐC) WHERE toLower(n.tính_chất) CONTAINS 'tinh thể' AND toLower(n.tính_chất) CONTAINS 'không màu' AND toLower(n.tính_chất) CONTAINS 'khó tan' RETURN n.tên_hoạt_chất, l.tên_loại",
    },
    {
        "question": "Xác định hoạt chất có tính chất [bột kết tinh trắng, đa hình, độ tan thấp]?",
        "query": "MATCH (n:HOẠT_CHẤT) WHERE toLower(n.tính_chất) CONTAINS 'bột' AND toLower(n.tính_chất) CONTAINS 'trắng' AND toLower(n.tính_chất) CONTAINS 'đa hình' RETURN n.tên_hoạt_chất, n.tên_latin, n.công_thức_hóa_học",
    }
]

# Cập nhật PREFIX với hướng dẫn xé nhỏ từ khóa cực kỳ quan trọng
PREFIX = """
Bạn là một chuyên gia về cơ sở dữ liệu đồ thị Neo4j. Nhiệm vụ của bạn là chuyển đổi câu hỏi Tiếng Việt thành truy vấn Cypher chính xác.

Cấu trúc cơ sở dữ liệu:
1. Node: HOẠT_CHẤT (tên_hoạt_chất, tên_latin, công_thức_hóa_học, mô_tả, bảo_quản, tính_chất)
2. Node: TIÊU_CHUẨN (định_lượng, định_tính, độ_hòa_tan, tạp_chất_và_độ_tinh_khiết, hàm_lượng_yêu_cầu)
   - Quan hệ: (:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(:TIÊU_CHUẨN)
3. Node: LOẠI_THUỐC (tên_loại)
   - Quan hệ: (:HOẠT_CHẤT)-[:THUỘC_NHÓM]->(:LOẠI_THUỐC)

HƯỚNG DẪN CHIẾN THUẬT QUAN TRỌNG:
- LUÔN SỬ DỤNG `toLower()`: Để tìm kiếm không phân biệt hoa thường.
- CHIẾN THUẬT XÉ NHỎ (KEYWORD SHREDDING): Đối với các mô tả trong ngoặc [ ], TUYỆT ĐỐI KHÔNG sử dụng nguyên văn cả chuỗi dài. Hãy tách thành các từ khóa đơn lẻ và nối bằng `AND`.
- ƯU TIÊN SỐ LIỆU: Nếu trong mô tả có số (nhiệt độ nóng chảy, điểm chảy), hãy đưa số đó vào truy vấn vì nó giúp định danh chính xác nhất.
- TRẢ VỀ ĐA TRƯỜNG: Khi hỏi về 'định tính' hoặc 'quy trình', hãy RETURN cả định_tính, định_lượng và độ_hòa_tan để đề phòng dữ liệu bị lệch cột.
"""

example_prompt = PromptTemplate.from_template("User input: {question}\nCypher query: {query}")

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
# 3. CHUẨN BỊ DỮ LIỆU
# ==============================================================================

data_test_1_hop =[]
data_test_2_hop=[]

# ==============================================================================
# 4. HÀM ĐÁNH GIÁ (EVALUATION FUNCTION)
# ==============================================================================

chen_smoothing = SmoothingFunction().method1

def run_evaluation(dataset, label_name):
    """
    Chạy đánh giá cho một bộ dữ liệu cụ thể.
    Trả về: (kết quả trung bình dict, danh sách logs chi tiết)
    """
    print(f"\n🚀 BẮT ĐẦU CHẠY THỬ NGHIỆM: {label_name.upper()} ({len(dataset)} mẫu)")
    
    total_bleu = 0
    total_rouge = 0
    total_meteor = 0
    local_logs = []

    for i, x in enumerate(dataset):
        print(f"\n🔹 [{label_name}] Câu hỏi {i+1}: {x['question']}")
        
        # Gọi Gemini Chain
        try:
            response = gemini_chain.invoke(x["question"])
            gemini_result = response.get('result', str(response))
        except Exception as e:
            gemini_result = "Không tìm thấy trong DB."
        
        if "I don't know" in str(gemini_result) or not gemini_result:
            gemini_result = "Không tìm thấy trong DB."
        
        print(f"✅ Trả lời: {gemini_result}")

        # Tính điểm
        reference = x["answer"]
        candidate = gemini_result
        ref_tokens = reference.split()
        cand_tokens = candidate.split()

        # BLEU
        b_score = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=chen_smoothing)
        
        # ROUGE
        try:
            if not candidate.strip(): r_score = 0
            else: r_score = rouge.get_scores(candidate, reference)[0]['rouge-l']['f']
        except: r_score = 0
        
        # METEOR
        try: m_score = meteor_score([ref_tokens], cand_tokens)
        except: m_score = 0

        total_bleu += b_score
        total_rouge += r_score
        total_meteor += m_score

        print(f"📊 Điểm: BLEU={b_score:.2f} | ROUGE={r_score:.2f} | METEOR={m_score:.2f}")

        local_logs.append({
            "type": label_name,
            "question": x["question"],
            "answer_ground_truth": reference,
            "answer_model": candidate,
            "scores": {"bleu": b_score, "rouge": r_score, "meteor": m_score}
        })
        
        time.sleep(1) # Delay nhẹ tránh rate limit

    # Tính trung bình
    n = len(dataset)
    if n > 0:
        avg_results = {
            "bleu": total_bleu / n,
            "rouge": total_rouge / n,
            "meteor": total_meteor / n,
            "count": n
        }
    else:
        avg_results = {"bleu": 0, "rouge": 0, "meteor": 0, "count": 0}

    return avg_results, local_logs

# ======================================================================
# LOAD DATASET TỪ FILE JSON
# ======================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "benchmark")

path_1hop = os.path.join(DATA_DIR, "1hop.json")
path_2hop = os.path.join(DATA_DIR, "2hop.json")

def load_json_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

data_test_1_hop = load_json_data(path_1hop)
data_test_2_hop = load_json_data(path_2hop)
MAX_QUESTIONS = 200

data_test_1_hop = load_json_data(path_1hop)[:MAX_QUESTIONS]
data_test_2_hop = load_json_data(path_2hop)[:MAX_QUESTIONS]

print(f"✅ 1-hop: chạy {len(data_test_1_hop)} câu hỏi")
print(f"✅ 2-hop: chạy {len(data_test_2_hop)} câu hỏi")

# ==============================================================================
# 5. CHẠY THỰC NGHIỆM VÀ GHI FILE
# ==============================================================================

results_dir = "results"
logs_dir = "logs"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

gemini_results_path = os.path.join(results_dir, "gemini_results.txt")
gemini_log_path = os.path.join(logs_dir, "gemini_log.json")

# --- CHẠY LẦN LƯỢT 2 BỘ DATA ---
avg_1_hop, logs_1_hop = run_evaluation(data_test_1_hop, "1-hop")
avg_2_hop, logs_2_hop = run_evaluation(data_test_2_hop, "2-hop")

# Tổng hợp log
full_logs = {
    "1_hop_data": logs_1_hop,
    "2_hop_data": logs_2_hop
}

# --- IN KẾT QUẢ RA MÀN HÌNH ---
print("\n" + "="*50)
print("🏆 TỔNG HỢP KẾT QUẢ BENCHMARK")
print("="*50)
print(f"🔹 1-HOP ({avg_1_hop['count']} mẫu):")
print(f"   BLEU: {avg_1_hop['bleu']:.4f} | ROUGE-L: {avg_1_hop['rouge']:.4f} | METEOR: {avg_1_hop['meteor']:.4f}")
print("-" * 50)
print(f"🔹 2-HOP ({avg_2_hop['count']} mẫu):")
print(f"   BLEU: {avg_2_hop['bleu']:.4f} | ROUGE-L: {avg_2_hop['rouge']:.4f} | METEOR: {avg_2_hop['meteor']:.4f}")
print("="*50)

# --- GHI FILE RESULTS TXT ---
with open(gemini_results_path, "w", encoding='utf-8') as f:
    f.write("BÁO CÁO KẾT QUẢ BENCHMARK (PHÂN LOẠI HOP)\n")
    f.write(f"Thời gian chạy: {time.ctime()}\n")
    f.write("==================================================\n\n")
    
    f.write(f"1. KẾT QUẢ 1-HOP (Số mẫu: {avg_1_hop['count']})\n")
    f.write(f"   - BLEU Score    : {avg_1_hop['bleu']:.4f}\n")
    f.write(f"   - ROUGE-L Score : {avg_1_hop['rouge']:.4f}\n")
    f.write(f"   - METEOR Score  : {avg_1_hop['meteor']:.4f}\n\n")
    
    f.write("--------------------------------------------------\n\n")

    f.write(f"2. KẾT QUẢ 2-HOP (Số mẫu: {avg_2_hop['count']})\n")
    f.write(f"   - BLEU Score    : {avg_2_hop['bleu']:.4f}\n")
    f.write(f"   - ROUGE-L Score : {avg_2_hop['rouge']:.4f}\n")
    f.write(f"   - METEOR Score  : {avg_2_hop['meteor']:.4f}\n")
    
    f.write("\n==================================================")

print(f"🎉 Đã lưu báo cáo tóm tắt vào: {gemini_results_path}")

# --- GHI FILE LOG JSON ---
with open(gemini_log_path, "w", encoding='utf-8') as f:
    json.dump(full_logs, f, ensure_ascii=False, indent=4)
print(f"🎉 Đã lưu log chi tiết vào: {gemini_log_path}")
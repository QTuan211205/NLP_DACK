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
MODEL_NAME = "gemini-2.5-flash" 

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
    {
        "question": "Công thức hóa học của Aspirin là gì?",
        "query": "MATCH (n:HOẠT_CHẤT) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('ASPIRIN') RETURN n.tên_hoạt_chất,n.công_thức_hóa_học",
    },
    {
        "question": "Mô tả chung về Paracetamol?",
        "query": "MATCH (n:HOẠT_CHẤT) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('PARACETAMOL') RETURN n.tên_hoạt_chất,n.mô_tả",
    },
    {
        "question": "Độ hòa tan của Glucose?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('GLUCOSE') RETURN n.tên_hoạt_chất,t.độ_hòa_tan",
    },
    {
        "question": "Cách định lượng hoạt chất Penicilamin?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('PENICILAMIN') RETURN n.tên_hoạt_chất, t.định_lượng",
    },
    {
        "question": "Phương pháp định tính viên nén Losartan Kali?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('LOSARTAN KALI') RETURN n.tên_hoạt_chất, t.định_tính",
    },
    {
        "question": "Điều kiện bảo quản của viên nén Propylthiouracil?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('PROPYLTHIOURACIL') RETURN n.tên_hoạt_chất, t.bảo_quản",
    },
    {
        "question": "Định lượng hoạt chất Ibuprofen như thế nào?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('IBUPROFEN') RETURN n.tên_hoạt_chất, t.định_lượng",
    },
    {
        "question": "Độ hòa tan của hoạt chất Artemether?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('ARTEMETHER') RETURN n.tên_hoạt_chất, t.độ_hòa_tan",
    },
    {
        "question": "Bảo quản Dexamethason Natri Phosphat như thế nào?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('DEXAMETHASON NATRI PHOSPHAT') RETURN n.tên_hoạt_chất, t.bảo_quản",
    },
    {
        "question": "Quy trình định lượng của Digoxin?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('DIGOXIN') RETURN n.tên_hoạt_chất, t.định_lượng",
    },
    {
        "question": "Natri Thiosulfat thuộc loại thuốc gì?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:THUỘC_LOẠI]->(l:LOẠI_THUỐC) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('NATRI THIOSULFAT') RETURN n.tên_hoạt_chất, l.tên_loại",
    },
    {
        "question": "Các phép thử tạp chất và độ tinh khiết của Abacavir Sulfat?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('ABACAVIR SULFAT') RETURN n.tên_hoạt_chất, t.tạp_chất_và_độ_tinh_khiết",
    },
    {
        "question": "Phương pháp định tính của hoạt chất Ceftazidim?",
        "query": "MATCH (n:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(t:TIÊU_CHUẨN) WHERE toLower(n.tên_hoạt_chất) CONTAINS toLower('CEFTAZIDIM') RETURN n.tên_hoạt_chất, t.định_tính",
    }
]

PREFIX = """
    You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query.
    My Database Schema (Tiếng Việt):
    1. Node: HOẠT_CHẤT (tên_hoạt_chất, tên_latin, công_thức_hóa_học, mô_tả, bảo_quản)
    2. Node: TIÊU_CHUẨN (định_lượng, định_tính, độ_hòa_tan, tạp_chất_và_độ_tinh_khiết, hàm_lượng_yêu_cầu)
       - Relation: (:HOẠT_CHẤT)-[:CÓ_TIÊU_CHUẨN]->(:TIÊU_CHUẨN)
    3. Node: LOẠI_THUỐC (tên_loại)
       - Relation: (:HOẠT_CHẤT)-[:THUỘC_NHÓM]->(:LOẠI_THUỐC)
    INSTRUCTIONS:
    - Use `toLower()` for case-insensitive search.
    - Use `CONTAINS` for fuzzy matching.
    - IMPORTANT: Use the EXACT Vietnamese property names.
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
    verbose=False,
    cypher_prompt=prompt,
    allow_dangerous_requests=True
)

# ==============================================================================
# 3. CHUẨN BỊ DỮ LIỆU
# ==============================================================================

data_test_1_hop = [
{ "question": "Công thức hóa học của ABACAVIR SULFAT là gì?", "answer": "Công thức hóa học của ABACAVIR SULFAT là C14H18N6O2.1." },
{ "question": "Tên Latin của ABACAVIR SULFAT là gì?", "answer": "Tên Latin của ABACAVIR SULFAT là Abacaviri sulfas." },
{ "question": "ABACAVIR SULFAT có độ hòa tan ra sao?", "answer": "ABACAVIR SULFAT có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACEBUTOLOL HYDROCLORID là gì?", "answer": "Công thức hóa học của ACEBUTOLOL HYDROCLORID là không có thông tin." },
{ "question": "Tên Latin của ACEBUTOLOL HYDROCLORID là gì?", "answer": "Tên Latin của ACEBUTOLOL HYDROCLORID là Acebutololi hydrochloridum." },
{ "question": "ACEBUTOLOL HYDROCLORID có độ hòa tan ra sao?", "answer": "ACEBUTOLOL HYDROCLORID có độ hòa tan (Phụ lục 11.4) Thiết bị: Kiểu cánh khuấy. Môi trường hòa tan: 900 ml nước. Tốc độ quay: 50 r/min. Thời gian: 30 min. Cách tiến hành: Dung dịch thử: Sau thời gian hòa tan quy định, lấy một phần dịch hòa tan, lọc. Pha loãng nếu cần bằng nước. Dung dịch chuẩn: Cân chính xác một lượng acebutolol hydroclorid chuẩn, hòa tan trong nước để thu được dung dịch có nồng độ acebutolol tương đương với nồng độ acebutolol của dung dịch thử. Đo độ hấp thụ (Phụ lục 4.1) của dung dịch thử và dung dịch chuẩn ở bước sóng 232 nm (Phụ lục 4.1). Tính hàm lượng acebutolol, C18H28ClNO4, dựa vào độ hấp thụ của dung dịch chuẩn, dung dịch thử và hàm lượng C18H28ClNO4 trong acebutolol hydroclorid chuẩn. Yêu cầu: Không ít hơn 80 %(Q) lượng acebutolol, C18H28ClNO4, so với lượng ghi trên nhãn được hòa tan trong 30 min.." },
{ "question": "Công thức hóa học của ACENOCOUMAROL là gì?", "answer": "Công thức hóa học của ACENOCOUMAROL là C19H15NO6." },
{ "question": "Tên Latin của ACENOCOUMAROL là gì?", "answer": "Tên Latin của ACENOCOUMAROL là Acenocoumarolum." },
{ "question": "ACENOCOUMAROL có độ hòa tan ra sao?", "answer": "ACENOCOUMAROL có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của VIÊN NÉN ACENOCOUMAROL là gì?", "answer": "Công thức hóa học của VIÊN NÉN ACENOCOUMAROL là C19H15NO6." },
{ "question": "Tên Latin của VIÊN NÉN ACENOCOUMAROL là gì?", "answer": "Tên Latin của VIÊN NÉN ACENOCOUMAROL là Tabellae Acenocoumaroli." },
{ "question": "VIÊN NÉN ACENOCOUMAROL có độ hòa tan ra sao?", "answer": "VIÊN NÉN ACENOCOUMAROL có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACETAZOLAMID là gì?", "answer": "Công thức hóa học của ACETAZOLAMID là C4H6N4O3S2." },
{ "question": "Tên Latin của ACETAZOLAMID là gì?", "answer": "Tên Latin của ACETAZOLAMID là Acetazolamidum." },
{ "question": "ACETAZOLAMID có độ hòa tan ra sao?", "answer": "ACETAZOLAMID có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của VIÊN NÉN ACETAZOLAMID là gì?", "answer": "Công thức hóa học của VIÊN NÉN ACETAZOLAMID là C4H6N4O3S2." },
{ "question": "Tên Latin của VIÊN NÉN ACETAZOLAMID là gì?", "answer": "Tên Latin của VIÊN NÉN ACETAZOLAMID là Tabulettae Acetazolamidi." },
{ "question": "VIÊN NÉN ACETAZOLAMID có độ hòa tan ra sao?", "answer": "VIÊN NÉN ACETAZOLAMID có độ hòa tan (Phụ lục 11.4) Thiết bị: Kiểu giỏ quay Môi trường hòa tan: 900 ml dung dịch acid hydrocloric 0,01 M (TT). Tốc độ quay: 100 r/min. Thời gian: 60 min. Cách tiến hành: Sau thời gian hòa tan quy định, lấy một phần dịch hòa tan và lọc, loại bỏ dịch lọc đầu. Pha loãng dịch lọc với dung dịch acid hydrocloric 0,01 M (TT) nếu cần. Đo độ hấp thụ của dung dịch thu được ở bước sóng cực đại khoảng 265 nm, trong cốc đo dày 1 cm, song song đo độ hấp thụ của dung dịch chuẩn acetazolamid có cùng nồng độ trong dung dịch acid hydrocloric 0,01 M (TT), dùng dung dịch acid hydrocloric 0,01 M (TT) làm mẫu trắng. Tính hàm lượng acetazolamid, C4H6N4O3S2, đã hòa tan trong mỗi viên dựa vào độ hấp thụ đo được của dung dịch chuẩn, dung dịch thử và hàm lượng C4H6N4O3S2 trong acetazolamid chuẩn. Yêu cầu: Không được ít hơn 75 % (Q) lượng acetazolamid, C4H6N4O3S2, so với lượng ghi trên nhãn được hòa tan trong 60 min.." },
{ "question": "Công thức hóa học của ACETYLCYSTEIN là gì?", "answer": "Công thức hóa học của ACETYLCYSTEIN là C5H9NO3S." },
{ "question": "Tên Latin của ACETYLCYSTEIN là gì?", "answer": "Tên Latin của ACETYLCYSTEIN là Acetylcystein." },
{ "question": "ACETYLCYSTEIN có độ hòa tan ra sao?", "answer": "ACETYLCYSTEIN có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của BỘT PHA HỖN DỊCH ACETYLCYSTEIN là gì?", "answer": "Công thức hóa học của BỘT PHA HỖN DỊCH ACETYLCYSTEIN là C5H9NO3S." },
{ "question": "Tên Latin của BỘT PHA HỖN DỊCH ACETYLCYSTEIN là gì?", "answer": "Tên Latin của BỘT PHA HỖN DỊCH ACETYLCYSTEIN là Pulveres Acetylcysteini." },
{ "question": "BỘT PHA HỖN DỊCH ACETYLCYSTEIN có độ hòa tan ra sao?", "answer": "BỘT PHA HỖN DỊCH ACETYLCYSTEIN có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của NANG ACETYLCYSTEIN là gì?", "answer": "Công thức hóa học của NANG ACETYLCYSTEIN là C5H9NO3S." },
{ "question": "Tên Latin của NANG ACETYLCYSTEIN là gì?", "answer": "Tên Latin của NANG ACETYLCYSTEIN là Capsulae Acetylcysteini." },
{ "question": "NANG ACETYLCYSTEIN có độ hòa tan ra sao?", "answer": "NANG ACETYLCYSTEIN có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACICLOVIR là gì?", "answer": "Công thức hóa học của ACICLOVIR là C8H11N5O3." },
{ "question": "Tên Latin của ACICLOVIR là gì?", "answer": "Tên Latin của ACICLOVIR là Aciclovirum." },
{ "question": "ACICLOVIR có độ hòa tan ra sao?", "answer": "ACICLOVIR có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của KEM ACICLOVIR là gì?", "answer": "Công thức hóa học của KEM ACICLOVIR là C8H11N5O3." },
{ "question": "Tên Latin của KEM ACICLOVIR là gì?", "answer": "Tên Latin của KEM ACICLOVIR là Cremoris Acicloviri." },
{ "question": "KEM ACICLOVIR có độ hòa tan ra sao?", "answer": "KEM ACICLOVIR có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của VIÊN NÉN ACICLOVIR là gì?", "answer": "Công thức hóa học của VIÊN NÉN ACICLOVIR là C8H11N5O3." },
{ "question": "Tên Latin của VIÊN NÉN ACICLOVIR là gì?", "answer": "Tên Latin của VIÊN NÉN ACICLOVIR là Tabellae Acicloviri." },
{ "question": "VIÊN NÉN ACICLOVIR có độ hòa tan ra sao?", "answer": "VIÊN NÉN ACICLOVIR có độ hòa tan (Phụ lục 11.4) Thiết bị: Kiểu cánh khuấy. Môi trường hòa tan: 900 ml dung dịch acid hydrocloric 0,1 M (TT). Tốc độ quay: 50 r/min. Thời gian: 45 min. Cách tiến hành: Lấy một phần dung dịch môi trường sau khi hòa tan, lọc, bỏ 20 ml dịch lọc đầu, pha loãng dịch lọc bằng dung dịch acid hydrocloric 0,1 M (TT) (nếu cần). Đo độ hấp thụ ánh sáng của dung dịch thu được ở bước sóng hấp thụ cực đại 255 nm (Phụ lục 4.1), cốc đo dày 1 cm, dùng dung dịch acid hydrocloric 0,1 M (TT) làm mẫu trắng. Tính hàm lượng aciclovir, C8H11N5O3, theo A (1 %, 1 cm), lấy 560 là giá trị A (1 %, 1 cm) ở cực đại 255 nm. Yêu cầu: Không được ít hơn 70 % (Q) lượng aciclovir, C8H11N5O3, so với lượng ghi trên nhãn được hòa tan trong 45 min.." },
{ "question": "Công thức hóa học của ACID ACETYLSALICYLIC là gì?", "answer": "Công thức hóa học của ACID ACETYLSALICYLIC là C9H8O4." },
{ "question": "Tên Latin của ACID ACETYLSALICYLIC là gì?", "answer": "Tên Latin của ACID ACETYLSALICYLIC là Acidum acetylsalicylicum." },
{ "question": "ACID ACETYLSALICYLIC có độ hòa tan ra sao?", "answer": "ACID ACETYLSALICYLIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của VIÊN NÉN ACID ACETYLSALICYLIC là gì?", "answer": "Công thức hóa học của VIÊN NÉN ACID ACETYLSALICYLIC là C9H8O4." },
{ "question": "Tên Latin của VIÊN NÉN ACID ACETYLSALICYLIC là gì?", "answer": "Tên Latin của VIÊN NÉN ACID ACETYLSALICYLIC là Tabellae Acidi acetylsalicylici." },
{ "question": "VIÊN NÉN ACID ACETYLSALICYLIC có độ hòa tan ra sao?", "answer": "VIÊN NÉN ACID ACETYLSALICYLIC có độ hòa tan (Phụ lục 11.4) Thiết bị: Kiểu giỏ quay. Môi trường hòa tan: 500 ml dung dịch đệm pH 4,5. Pha dung dịch đệm pH 4,5: Hòa tan 29,9 g natri acetat (TT) trong nước, thêm 16,6 ml acid acetic băng (TT) và thêm nước vừa đủ 10 L. Tốc độ quay: 50 r/min. Thời gian: 45 min. Cách tiến hành: Lấy một lượng dung dịch hòa tan, lọc, bỏ 10 ml dịch lọc đầu. Đo độ hấp thụ ánh sáng ngay lập tức ở bước sóng 265 nm (Phụ lục 4.1) (nếu cần pha loãng dịch lọc với môi trường hòa tan để có nồng độ thích hợp), so với mẫu trắng là môi trường hòa tan. Song song đo độ hấp thụ ánh sáng của dung dịch acid acetylsalicylic chuẩn có nồng độ tương đương được pha trong môi trường hòa tan. Từ hàm lượng acid acetylsalicylic chuẩn, tính hàm lượng acid acetylsalicylic, C9H8O4, có trong dung dịch mẫu thử đã hòa tan. Yêu cầu: Không được ít hơn 70 % (Q) hàm lượng acid acetylsalicylic, C9H8O4, so với hàm lượng ghi trên nhãn được hòa tan trong 45 min. Giới hạn acid salicylic tự do Không được quá 3,0 %. Cân một lượng bột viên tương ứng với 0,2 g acid acetylsalicylic, lắc với 4 ml ethanol 96 % (TT) và pha loãng với nước đến 100 ml ở nhiệt độ không quá 10 °C. Lọc ngay bằng giấy lọc và lấy 50 ml dịch lọc vào ống so màu Nessler, thêm vào 1 ml dung dịch phèn sắt amoni 0,2 % (TT) mới pha, trộn đều và để yên trong 1 phút. Dung dịch này không được có màu tím đậm hơn màu của dung dịch mẫu [gồm 1 ml dung dịch phèn sắt amoni 0,2 % (TT) mới pha và hỗn hợp của 3 ml dung dịch acid salicylic 0,010 % (TT) mới pha, 2 ml ethanol 96 % (TT) và nước vừa đủ 50 ml].." },
{ "question": "Công thức hóa học của VIÊN NÉN BAO TAN TRONG RUỘT ACID là gì?", "answer": "Công thức hóa học của VIÊN NÉN BAO TAN TRONG RUỘT ACID là C9H8O4." },
{ "question": "Tên Latin của VIÊN NÉN BAO TAN TRONG RUỘT ACID là gì?", "answer": "Tên Latin của VIÊN NÉN BAO TAN TRONG RUỘT ACID là Tabellae Acidi acetylsalicylici." },
{ "question": "VIÊN NÉN BAO TAN TRONG RUỘT ACID có độ hòa tan ra sao?", "answer": "VIÊN NÉN BAO TAN TRONG RUỘT ACID có độ hòa tan (Phụ lục 11.4) Giai đoạn trong môi trường acid Thiết bị: Kiểu giỏ quay. Môi trường hòa tan: 1000 ml dung dịch acid hydrocloric 0,1 M (TT). Tốc độ quay: 100 r/min. Thời gian: 2 h. Cách tiến hành: Sau thời gian quy định, lấy một phần dịch hòa tan, lọc. Pha loãng dịch lọc với môi trường hòa tan (nếu cần) và đo độ hấp thụ (Phụ lục 4.1) của dung dịch thu được ở bước sóng 276 nm với mẫu trắng là môi trường hòa tan. So sánh với dung dịch acid acetylsalicylic chuẩn có nồng độ tương đương pha trong cùng dung môi. Tính hàm lượng acid acetylsalicylic, C9H8O4, hòa tan trong mỗi viên dựa vào độ hấp thụ của dung dịch chuẩn, dung dịch thử và hàm lượng C9H8O4 trong acid acetylsalicylic chuẩn. Yêu cầu: Không được quá 10 % lượng acid acetylsalicylic, C9H8O4, so với lượng ghi trên nhãn hòa tan trong 2 h. Giai đoạn trong môi trường đệm Tiếp tục ngay sau khi kết thúc giai đoạn trong môi trường acid trên cùng mẫu thử. Thiết bị: Kiểu giỏ quay. Môi trường hòa tan: Đệm phosphat hỗn hợp pH 6.8 (TT). Tốc độ quay: 100 r/min. Thời gian: 45 min. Cách tiến hành: Thay thế dung dịch acid hydrocloric 0,1M trong bình thử độ hòa tan bằng 900 ml đệm phosphat hỗn hợp pH 6,8 (TT) đã làm nóng trước đến 37 °C ± 0.5 °C. Sau thời gian hòa tan quy định, lấy một phần dịch hòa tan, lọc. Pha loãng dịch lọc với môi trường hòa tan (nếu cần) và đo ngay độ hấp thụ (Phụ lục 4.1) của dung dịch thu được (dung dịch thử) ở bước sóng 265 nm với mẫu trắng là đệm phosphat hỗn hợp pH 6.8 (TT). So sánh với dung dịch acid acetylsalicylic chuẩn có nồng độ tương đương pha trong cùng dung môi. Tính hàm lượng acid acetylsalicylic, C9H8O4, hòa tan trong mỗi viên dựa vào độ hấp thụ của dung dịch chuẩn, dung dịch thử và hàm lượng C9H8O4 trong acid acetylsalicylic chuẩn. Yêu cầu: Không ít hơn 70 % (Q) lượng acid acetylsalicylic so với lượng ghi trên nhãn được hòa tan trong 45 min. Giới hạn acid salicylic tự do Không được quá 3,0 %. Phương pháp sắc ký lỏng (Phụ lục 5.3). Pha động: Acetonitril - dung dịch natri dihydrophosphat 0,05 M được chỉnh đến pH 2,0 bằng acid phosphoric (1:3).." },
{ "question": "Công thức hóa học của VIÊN NÉN ASPIRIN VÀ CAFEIN là gì?", "answer": "Công thức hóa học của VIÊN NÉN ASPIRIN VÀ CAFEIN là C9H8O4." },
{ "question": "Tên Latin của VIÊN NÉN ASPIRIN VÀ CAFEIN là gì?", "answer": "Tên Latin của VIÊN NÉN ASPIRIN VÀ CAFEIN là Tabellae Aspirini et Coffeini." },
{ "question": "VIÊN NÉN ASPIRIN VÀ CAFEIN có độ hòa tan ra sao?", "answer": "VIÊN NÉN ASPIRIN VÀ CAFEIN có độ hòa tan (Phụ lục 11.4) Thiết bị: Kiểu cánh khuấy. Môi trường hòa tan: 500 ml dung dịch đệm acetat 0,05 M pH 4,5 được chuẩn bị như sau: Hòa tan 2,99 g natri acetat (TT) và 1,66 ml acid acetic băng (TT) trong nước và thêm nước vừa đủ 1000 ml. Tốc độ quay: 50 r/min. Thời gian: 45 min. Cách tiến hành: Sau thời gian hòa tan quy định, lấy một phần dịch hòa tan, lọc, bỏ 20 ml dịch lọc đầu. Tiến hành định lượng aspirin và cafein hòa tan bằng phương pháp sắc ký lỏng (Phụ lục 5.3) với pha động và điều kiện sắc ký như phần Định lượng. Chuẩn bị dung dịch aspirin chuẩn và cafein chuẩn trong môi trường hòa tan có nồng độ tương đương với nồng độ aspirin và cafein tương ứng trong dung dịch thử. Yêu cầu: Không ít hơn 75 % (Q) lượng aspirin, C9H8O4, so với lượng ghi trên nhãn được hòa tan trong 45 min. Không ít hơn 75 % (Q) lượng cafein, C8H10N4O2, so với lượng ghi trên nhãn được hòa tan trong 45 min.." },
{ "question": "Công thức hóa học của ACID AMINOCAPROIC là gì?", "answer": "Công thức hóa học của ACID AMINOCAPROIC là C6H13NO2." },
{ "question": "Tên Latin của ACID AMINOCAPROIC là gì?", "answer": "Tên Latin của ACID AMINOCAPROIC là Acidum Aminocaproicum." },
{ "question": "ACID AMINOCAPROIC có độ hòa tan ra sao?", "answer": "ACID AMINOCAPROIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID ASCORBIC là gì?", "answer": "Công thức hóa học của ACID ASCORBIC là C6H8O6." },
{ "question": "Tên Latin của ACID ASCORBIC là gì?", "answer": "Tên Latin của ACID ASCORBIC là Acidum Ascorbicum." },
{ "question": "ACID ASCORBIC có độ hòa tan ra sao?", "answer": "ACID ASCORBIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của THUỐC TIÊM ACID ASCORBIC là gì?", "answer": "Công thức hóa học của THUỐC TIÊM ACID ASCORBIC là C6H8O6." },
{ "question": "Tên Latin của THUỐC TIÊM ACID ASCORBIC là gì?", "answer": "Tên Latin của THUỐC TIÊM ACID ASCORBIC là Injectio Acidi ascorbici." },
{ "question": "THUỐC TIÊM ACID ASCORBIC có độ hòa tan ra sao?", "answer": "THUỐC TIÊM ACID ASCORBIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của VIÊN NÉN ACID ASCORBIC là gì?", "answer": "Công thức hóa học của VIÊN NÉN ACID ASCORBIC là C6H8O6." },
{ "question": "Tên Latin của VIÊN NÉN ACID ASCORBIC là gì?", "answer": "Tên Latin của VIÊN NÉN ACID ASCORBIC là Tabellae Acidi ascorbici." },
{ "question": "VIÊN NÉN ACID ASCORBIC có độ hòa tan ra sao?", "answer": "VIÊN NÉN ACID ASCORBIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID BENZOIC là gì?", "answer": "Công thức hóa học của ACID BENZOIC là C7H6O2." },
{ "question": "Tên Latin của ACID BENZOIC là gì?", "answer": "Tên Latin của ACID BENZOIC là Acidum benzoicum." },
{ "question": "ACID BENZOIC có độ hòa tan ra sao?", "answer": "ACID BENZOIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của THUỐC MỠ BENZOSALICYLIC là gì?", "answer": "Công thức hóa học của THUỐC MỠ BENZOSALICYLIC là C7H6O2." },
{ "question": "Tên Latin của THUỐC MỠ BENZOSALICYLIC là gì?", "answer": "Tên Latin của THUỐC MỠ BENZOSALICYLIC là Unguentum Benzosalicylici." },
{ "question": "THUỐC MỠ BENZOSALICYLIC có độ hòa tan ra sao?", "answer": "THUỐC MỠ BENZOSALICYLIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID BORIC là gì?", "answer": "Công thức hóa học của ACID BORIC là không có thông tin." },
{ "question": "Tên Latin của ACID BORIC là gì?", "answer": "Tên Latin của ACID BORIC là Acidum boricum." },
{ "question": "ACID BORIC có độ hòa tan ra sao?", "answer": "ACID BORIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của DUNG DỊCH ACID BORIC 3 % là gì?", "answer": "Công thức hóa học của DUNG DỊCH ACID BORIC 3 % là không có thông tin." },
{ "question": "Tên Latin của DUNG DỊCH ACID BORIC 3 % là gì?", "answer": "Tên Latin của DUNG DỊCH ACID BORIC 3 % là Solutio Acidi borici 3 %." },
{ "question": "DUNG DỊCH ACID BORIC 3 % có độ hòa tan ra sao?", "answer": "DUNG DỊCH ACID BORIC 3 % có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của THUỐC MỠ ACID BORIC 10 % là gì?", "answer": "Công thức hóa học của THUỐC MỠ ACID BORIC 10 % là không có thông tin." },
{ "question": "Tên Latin của THUỐC MỠ ACID BORIC 10 % là gì?", "answer": "Tên Latin của THUỐC MỠ ACID BORIC 10 % là Unguentum Acidi borici 10 %." },
{ "question": "THUỐC MỠ ACID BORIC 10 % có độ hòa tan ra sao?", "answer": "THUỐC MỠ ACID BORIC 10 % có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID CITRIC NGẬM MỘT PHÂN TỬ NƯỚC là gì?", "answer": "Công thức hóa học của ACID CITRIC NGẬM MỘT PHÂN TỬ NƯỚC là C6H8O7." },
{ "question": "Tên Latin của ACID CITRIC NGẬM MỘT PHÂN TỬ NƯỚC là gì?", "answer": "Tên Latin của ACID CITRIC NGẬM MỘT PHÂN TỬ NƯỚC là Acidum citricum monohydricum." },
{ "question": "ACID CITRIC NGẬM MỘT PHÂN TỬ NƯỚC có độ hòa tan ra sao?", "answer": "ACID CITRIC NGẬM MỘT PHÂN TỬ NƯỚC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID FOLIC là gì?", "answer": "Công thức hóa học của ACID FOLIC là C19H19N7O6." },
{ "question": "Tên Latin của ACID FOLIC là gì?", "answer": "Tên Latin của ACID FOLIC là Acidum folicum." },
{ "question": "ACID FOLIC có độ hòa tan ra sao?", "answer": "ACID FOLIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của VIÊN NÉN ACID FOLIC là gì?", "answer": "Công thức hóa học của VIÊN NÉN ACID FOLIC là C19H19N7O6." },
{ "question": "Tên Latin của VIÊN NÉN ACID FOLIC là gì?", "answer": "Tên Latin của VIÊN NÉN ACID FOLIC là Tabellae Acidi folici." },
{ "question": "VIÊN NÉN ACID FOLIC có độ hòa tan ra sao?", "answer": "VIÊN NÉN ACID FOLIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID HYDROCLORIC là gì?", "answer": "Công thức hóa học của ACID HYDROCLORIC là không có thông tin." },
{ "question": "Tên Latin của ACID HYDROCLORIC là gì?", "answer": "Tên Latin của ACID HYDROCLORIC là Acidum hydrochloricum." },
{ "question": "ACID HYDROCLORIC có độ hòa tan ra sao?", "answer": "ACID HYDROCLORIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID HYDROCLORIC LOÃNG là gì?", "answer": "Công thức hóa học của ACID HYDROCLORIC LOÃNG là không có thông tin." },
{ "question": "Tên Latin của ACID HYDROCLORIC LOÃNG là gì?", "answer": "Tên Latin của ACID HYDROCLORIC LOÃNG là Acidum hydrochloricum dilutum." },
{ "question": "ACID HYDROCLORIC LOÃNG có độ hòa tan ra sao?", "answer": "ACID HYDROCLORIC LOÃNG có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID MEFENAMIC là gì?", "answer": "Công thức hóa học của ACID MEFENAMIC là C15H15NO2." },
{ "question": "Tên Latin của ACID MEFENAMIC là gì?", "answer": "Tên Latin của ACID MEFENAMIC là Acid mefenamic." },
{ "question": "ACID MEFENAMIC có độ hòa tan ra sao?", "answer": "ACID MEFENAMIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của VIÊN NÉN ACID MEFENAMIC là gì?", "answer": "Công thức hóa học của VIÊN NÉN ACID MEFENAMIC là C15H15NO2." },
{ "question": "Tên Latin của VIÊN NÉN ACID MEFENAMIC là gì?", "answer": "Tên Latin của VIÊN NÉN ACID MEFENAMIC là Tabellae Acidi mefenamici." },
{ "question": "VIÊN NÉN ACID MEFENAMIC có độ hòa tan ra sao?", "answer": "VIÊN NÉN ACID MEFENAMIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID METHACRYLIC VÀ METHYL là gì?", "answer": "Công thức hóa học của ACID METHACRYLIC VÀ METHYL là không có thông tin." },
{ "question": "Tên Latin của ACID METHACRYLIC VÀ METHYL là gì?", "answer": "Tên Latin của ACID METHACRYLIC VÀ METHYL là Acidi methacrylici et methylis methacrylatis polymerisatum (1 : 2)." },
{ "question": "ACID METHACRYLIC VÀ METHYL có độ hòa tan ra sao?", "answer": "ACID METHACRYLIC VÀ METHYL có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID NALIDIXIC là gì?", "answer": "Công thức hóa học của ACID NALIDIXIC là C12H12N2O3." },
{ "question": "Tên Latin của ACID NALIDIXIC là gì?", "answer": "Tên Latin của ACID NALIDIXIC là Acid nalidixic." },
{ "question": "ACID NALIDIXIC có độ hòa tan ra sao?", "answer": "ACID NALIDIXIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của VIÊN NÉN ACID NALIDIXIC là gì?", "answer": "Công thức hóa học của VIÊN NÉN ACID NALIDIXIC là C12H12N2O3." },
{ "question": "Tên Latin của VIÊN NÉN ACID NALIDIXIC là gì?", "answer": "Tên Latin của VIÊN NÉN ACID NALIDIXIC là Tabellae Acidi nalidixici." },
{ "question": "VIÊN NÉN ACID NALIDIXIC có độ hòa tan ra sao?", "answer": "VIÊN NÉN ACID NALIDIXIC có độ hòa tan (Phụ lục 11.4) Thiết bị: Kiểu cánh khuấy. Môi trường hòa tan: 900 ml dung dịch đệm methanol-phosphat được pha như sau: Trộn 2,3 thể tích dung dịch natri hydroxyd 0,2 M (TT) với 2,5 thể tích dung dịch kali dihydrophosphat 0,2 M (TT) và 2,0 thể tích methanol. Pha loãng thành 10 thể tích với nước, chỉnh đến pH 8,6 bằng dung dịch natri hydroxyd 0,1 M (TT). Tốc độ quay: 60 r/min. Thời gian: 45 phút. Cách tiến hành: Lấy một phần dung dịch môi trường sau khi hòa tan, lọc, bỏ 20 ml dịch lọc đầu. Pha loãng dịch lọc bằng môi trường hòa tan (nếu cần). Đo độ hấp thụ ánh sáng của dung dịch thử ở bước sóng hấp thụ cực đại 334 nm (Phụ lục 4.1), cốc đo dày 1 cm, dùng dung dịch môi trường hòa tan làm mẫu trắng. Tính hàm lượng acid nalidixic, C12H12N2O3, đã hòa tan trong mỗi viên theo A (1 %, 1 cm), lấy 494 là giá trị A (1 %, 1 cm) ở cực đại 334 nm. Yêu cầu: Không được ít hơn 70 % (Q) lượng acid nalidixic, C12H12N2O3, so với lượng ghi trên nhãn được hòa tan trong 45 phút.." },
{ "question": "Công thức hóa học của ACID NICOTINIC là gì?", "answer": "Công thức hóa học của ACID NICOTINIC là C6H5NO2." },
{ "question": "Tên Latin của ACID NICOTINIC là gì?", "answer": "Tên Latin của ACID NICOTINIC là Acidum nicotinicum." },
{ "question": "ACID NICOTINIC có độ hòa tan ra sao?", "answer": "ACID NICOTINIC có độ hòa tan (Phụ lục 11.4) Thiết bị: Kiểu cánh khuấy. Môi trường hòa tan: Dùng 40 ml ethanol (TT) và thêm dung dịch đệm phosphat pH 8,0 tới 800 ml. Dung dịch đệm phosphat pH 8,0: Hòa tan 5,59 g dikali hydrophosphat (TT) và 0,41 g kali dihydrophosphat (TT) trong 1000 ml nước. Tốc độ quay: 75 r/min. Thời gian: 45 min. Cách tiến hành: Dung dịch thử: Sau thời gian hòa tan qui định, lấy một phần dịch hòa tan, lọc. Pha loãng dịch lọc với dung dịch đệm phosphat pH 8,0 (TT) để thu được dung dịch có nồng độ 10 µg/ml. Dung dịch chuẩn: Cân chính xác khoảng 20 mg acid mefenamic chuẩn vào bình định mức 100 ml, thêm 5 ml ethanol (TT) để hòa tan, thêm dung dịch đệm phosphat pH 8,0 đến vạch, trộn đều. Pha loãng dung dịch thu được với dung dịch đệm phosphat pH 8,0 để thu được dung dịch có nồng độ 10 µg/ml. Đo độ hấp thụ của dung dịch thử và dung dịch chuẩn ở 286 nm (Phụ lục 4.1), dùng dung dịch đệm phosphat pH 8,0 làm mẫu trắng. Tính lượng acid mefenamic, C15H15NO2, được hòa tan trong mỗi viên. Yêu cầu: Không được ít hơn 60 % (Q) lượng acid mefenamic, C15H15NO2, so với lượng ghi trên nhãn được hòa tan trong 45 min.." },
{ "question": "Công thức hóa học của ACID METHACRYLIC VÀ ETHYL ACRYLAT là gì?", "answer": "Công thức hóa học của ACID METHACRYLIC VÀ ETHYL ACRYLAT là không có thông tin." },
{ "question": "Tên Latin của ACID METHACRYLIC VÀ ETHYL ACRYLAT là gì?", "answer": "Tên Latin của ACID METHACRYLIC VÀ ETHYL ACRYLAT là Acidi methacrylici et ethylis acrylatis polymerisatint (1 : 1)." },
{ "question": "ACID METHACRYLIC VÀ ETHYL ACRYLAT có độ hòa tan ra sao?", "answer": "ACID METHACRYLIC VÀ ETHYL ACRYLAT có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của DỊCH PHÂN TÁN 30 % CỦA ACID METHACRYLIC là gì?", "answer": "Công thức hóa học của DỊCH PHÂN TÁN 30 % CỦA ACID METHACRYLIC là không có thông tin." },
{ "question": "Tên Latin của DỊCH PHÂN TÁN 30 % CỦA ACID METHACRYLIC là gì?", "answer": "Tên Latin của DỊCH PHÂN TÁN 30 % CỦA ACID METHACRYLIC là Acidi methacrylici et ethyls acrylatis polymerisati 1 : 1 dispersio 30 per centum." },
{ "question": "DỊCH PHÂN TÁN 30 % CỦA ACID METHACRYLIC có độ hòa tan ra sao?", "answer": "DỊCH PHÂN TÁN 30 % CỦA ACID METHACRYLIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID METHACRYLIC VÀ METHYL là gì?", "answer": "Công thức hóa học của ACID METHACRYLIC VÀ METHYL là không có thông tin." },
{ "question": "Tên Latin của ACID METHACRYLIC VÀ METHYL là gì?", "answer": "Tên Latin của ACID METHACRYLIC VÀ METHYL là Acidi methacrylici et methylis methacrylatis polymerisatum (I : I)." },
{ "question": "ACID METHACRYLIC VÀ METHYL có độ hòa tan ra sao?", "answer": "ACID METHACRYLIC VÀ METHYL có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID SALICYLIC là gì?", "answer": "Công thức hóa học của ACID SALICYLIC là C7H6O3." },
{ "question": "Tên Latin của ACID SALICYLIC là gì?", "answer": "Tên Latin của ACID SALICYLIC là Acidum salicylicium." },
{ "question": "ACID SALICYLIC có độ hòa tan ra sao?", "answer": "ACID SALICYLIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ACID TRANEXAMIC là gì?", "answer": "Công thức hóa học của ACID TRANEXAMIC là C8H15NO2." },
{ "question": "Tên Latin của ACID TRANEXAMIC là gì?", "answer": "Tên Latin của ACID TRANEXAMIC là Acidum tranexamicum." },
{ "question": "ACID TRANEXAMIC có độ hòa tan ra sao?", "answer": "ACID TRANEXAMIC có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của NANG ACID TRANEXAMIC là gì?", "answer": "Công thức hóa học của NANG ACID TRANEXAMIC là C8H15NO2." },
{ "question": "Tên Latin của NANG ACID TRANEXAMIC là gì?", "answer": "Tên Latin của NANG ACID TRANEXAMIC là Capsulae Acidi tranexamici." },
{ "question": "NANG ACID TRANEXAMIC có độ hòa tan ra sao?", "answer": "NANG ACID TRANEXAMIC có độ hòa tan Thiết bị: Kiểu cánh khuấy. Môi trường hòa tan: 900 ml nước. Tốc độ quay: 50 r/min. Thời gian: 15 min. Cách tiến hành: Phương pháp sắc ký lỏng (Phụ lục 5.3) với các điều kiện sắc ký như mô tả trong phần Định lượng. Dung dịch thử: Sau thời gian hòa tan qui định, lấy một phần dịch hòa tan, lọc. Pha loãng dịch lọc với nước để được dung dịch có nồng độ acid tranexamic khoảng 0,56 mg/ml. Dung dịch chuẩn: Cân chính xác khoảng 56 mg acid tranexamic chuẩn, hòa tan trong nước và thêm nước vừa đủ 100,0 ml. Tiến hành sắc ký lần lượt đối với dung dịch chuẩn và dung dịch thử với thể tích tiêm là 50 µl. Tính hàm lượng acid tranexamic hòa tan trong mỗi nang dựa vào diện tích pic của acid tranexamic trên sắc ký đồ thu được từ dung dịch thử, dung dịch chuẩn và hàm lượng C8H15NO2 của acid tranexamic chuẩn. Yêu cầu: Không ít hơn 80 % (Q) lượng acid tranexamic, C8H15NO2, so với lượng ghi trên nhãn được hòa tan trong 15 min.." },
{ "question": "Công thức hóa học của VIÊN NÉN ACID TRANEXAMIC là gì?", "answer": "Công thức hóa học của VIÊN NÉN ACID TRANEXAMIC là C8H15NO2." },
{ "question": "Tên Latin của VIÊN NÉN ACID TRANEXAMIC là gì?", "answer": "Tên Latin của VIÊN NÉN ACID TRANEXAMIC là Tabellae Acidi tranexamici." },
{ "question": "VIÊN NÉN ACID TRANEXAMIC có độ hòa tan ra sao?", "answer": "VIÊN NÉN ACID TRANEXAMIC có độ hòa tan (Phụ lục 11.4) Thiết bị: Kiểu cánh khuấy. Môi trường hòa tan: 900 ml nước. Tốc độ quay: 50 r/min. Thời gian: 15 min. Cách tiến hành: Tiến hành phương pháp sắc ký lỏng với các điều kiện sắc ký như mô tả ở mục Định lượng (Phụ lục 5.3). Dung dịch thử: Sau thời gian hòa tan qui định, lấy một phần dịch hòa tan, lọc. Pha loãng dịch lọc với nước để được dung dịch có nồng độ acid tranexamic khoảng 0,56 mg/ml. Dung dịch chuẩn: Cân chính xác khoảng 56 mg acid tranexamic chuẩn hòa tan trong nước và thêm nước vừa đủ 100,0 ml. Tiến hành sắc ký lần lượt đối với dung dịch chuẩn và dung dịch thử với thể tích tiêm là 50 µl. Tính hàm lượng acid tranexamic hòa tan trong mỗi viên dựa vào diện tích pic của acid tranexamic trên sắc ký đồ thu được từ dung dịch thử, dung dịch chuẩn và hàm lượng C8H15NO2 trong acid tranexamic chuẩn. Yêu cầu: Không ít hơn 80 % (Q) lượng acid tranexamic, C8H15NO2, so với lượng ghi trên nhãn được hòa tan trong 15 min.." },
{ "question": "Công thức hóa học của ADRENALIN là gì?", "answer": "Công thức hóa học của ADRENALIN là C9H13NO3." },
{ "question": "Tên Latin của ADRENALIN là gì?", "answer": "Tên Latin của ADRENALIN là Adrenalinum." },
{ "question": "ADRENALIN có độ hòa tan ra sao?", "answer": "ADRENALIN có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ADRENALIN ACID TARTRAT là gì?", "answer": "Công thức hóa học của ADRENALIN ACID TARTRAT là C9H13NO3.C4H6O6." },
{ "question": "Tên Latin của ADRENALIN ACID TARTRAT là gì?", "answer": "Tên Latin của ADRENALIN ACID TARTRAT là Adrenalinum Acidum Tartratis." },
{ "question": "ADRENALIN ACID TARTRAT có độ hòa tan ra sao?", "answer": "ADRENALIN ACID TARTRAT có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của THUỐC TIÊM ADRENALIN là gì?", "answer": "Công thức hóa học của THUỐC TIÊM ADRENALIN là C9H13NO3." },
{ "question": "Tên Latin của THUỐC TIÊM ADRENALIN là gì?", "answer": "Tên Latin của THUỐC TIÊM ADRENALIN là Injectio Adrenalini." },
{ "question": "THUỐC TIÊM ADRENALIN có độ hòa tan ra sao?", "answer": "THUỐC TIÊM ADRENALIN có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của ALBENDAZOL là gì?", "answer": "Công thức hóa học của ALBENDAZOL là C12H15N3O2S." },
{ "question": "Tên Latin của ALBENDAZOL là gì?", "answer": "Tên Latin của ALBENDAZOL là Albendazolum." },
{ "question": "ALBENDAZOL có độ hòa tan ra sao?", "answer": "ALBENDAZOL có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của VIÊN NÉN ALBENDAZOL là gì?", "answer": "Công thức hóa học của VIÊN NÉN ALBENDAZOL là C12H15N3O2S." },
{ "question": "Tên Latin của VIÊN NÉN ALBENDAZOL là gì?", "answer": "Tên Latin của VIÊN NÉN ALBENDAZOL là Tabellae Albendazoli." },
{ "question": "VIÊN NÉN ALBENDAZOL có độ hòa tan ra sao?", "answer": "VIÊN NÉN ALBENDAZOL có độ hòa tan (Phụ lục 11.4) Thiết bị: Kiểu cánh khuấy. Môi trường hòa tan: 900 ml dung dịch acid hydrocloric 0,1 M (TT). Tốc độ quay: 75 r/min. Thời gian: 30 min. Dung dịch methanol acid: Lấy 50 ml methanol (TT) cho vào bình định mức 100 ml, thêm 2 ml acid hydrocloric (TT), pha loãng vừa đủ với methanol (TT) đến vạch. Dung dịch chuẩn: Cân chính xác khoảng 90 mg albendazol chuẩn cho vào bình định mức 250 ml, thêm 10 ml dung dịch methanol acid, lắc để hòa tan. Pha loãng với dung dịch acid hydrocloric 0,1 M (TT) vừa đủ đến vạch và lắc đều. Lấy 5,0 ml dung dịch này cho vào bình định mức 200 ml, pha loãng với dung dịch natri hydroxyd 0,1 M (TT) vừa đủ đến vạch, lắc đều. Cách tiến hành: Lấy một phần dung dịch môi trường sau khi hòa tan, lọc, bỏ 20 ml dịch lọc đầu. Pha loãng dịch lọc với dung dịch natri hydroxyd 0,1 M (TT) để thu được dung dịch có nồng độ tương đương với dung dịch chuẩn. Đo độ hấp thụ của dung dịch này và dung dịch chuẩn ở bước sóng cực đại khoảng 308 nm và cực tiểu khoảng 350 nm (Phụ lục 4.1), cốc đo dày 1 cm. Dùng dung dịch natri hydroxyd 0,1 M (TT) làm mẫu trắng. Tính hàm lượng albendazol, C12H15N3O2S, đã hòa tan theo cách tính trong phần Định lượng. Yêu cầu: Không được ít hơn 80 % (Q) lượng albendazol, C12H15N3O2S, so với lượng ghi trên nhãn được hòa tan trong 30 min.." },
{ "question": "Công thức hóa học của ALIMEMAZIN TARTRAT là gì?", "answer": "Công thức hóa học của ALIMEMAZIN TARTRAT là C18H22N2S.C4H6O6." },
{ "question": "Tên Latin của ALIMEMAZIN TARTRAT là gì?", "answer": "Tên Latin của ALIMEMAZIN TARTRAT là Alimemazini tartras." },
{ "question": "ALIMEMAZIN TARTRAT có độ hòa tan ra sao?", "answer": "ALIMEMAZIN TARTRAT có độ hòa tan không có thông tin." },
{ "question": "Công thức hóa học của VIÊN NÉN ALIMEMAZIN là gì?", "answer": "Công thức hóa học của VIÊN NÉN ALIMEMAZIN là C18H22N2S.C4H6O6." },
{ "question": "Tên Latin của VIÊN NÉN ALIMEMAZIN là gì?", "answer": "Tên Latin của VIÊN NÉN ALIMEMAZIN là Tabellae Alimemazini." },
{ "question": "VIÊN NÉN ALIMEMAZIN có độ hòa tan ra sao?", "answer": "VIÊN NÉN ALIMEMAZIN có độ hòa tan Thiết bị: Kiểu giỏ quay. Môi trường hòa tan: 500 ml dung dịch acid hydrocloric 0,01 M (TT). Tốc độ quay: 100 r/min. Thời gian: 45 min. Cách tiến hành: Sau thời gian hòa tan quy định, hút dịch hòa tan, lọc. Pha loãng nếu cần với môi trường hòa tan. Đo độ hấp thụ của dung dịch thử ở bước sóng cực đại khoảng 254 nm (Phụ lục 4.1), sử dụng cốc đo dày 1 cm, mẫu trắng là môi trường hòa tan, so sánh với dung dịch chuẩn alimemazin tartrat có cùng nồng độ pha trong môi trường hòa tan. Yêu cầu: Không được ít hơn 75 % (Q) lượng alimemazin tartrat so với lượng ghi trên nhãn được hòa tan trong 45 min.." },
]

data_test_2_hop = [
{ "question": "Hoạt chất có công thức hóa học C14H18N6O2.1 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C14H18N6O2.1 là ABACAVIR SULFAT, có tên Latin là Abacaviri sulfas." },
{ "question": "Hoạt chất có tên Latin là Abacaviri sulfas có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Abacaviri sulfas là ABACAVIR SULFAT, có công thức hóa học C14H18N6O2.1." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là ACEBUTOLOL HYDROCLORID, có tên Latin là Acebutololi hydrochloridum." },
{ "question": "Hoạt chất có tên Latin là Acebutololi hydrochloridum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acebutololi hydrochloridum là ACEBUTOLOL HYDROCLORID, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học C19H15NO6 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C19H15NO6 là ACENOCOUMAROL, có tên Latin là Acenocoumarolum." },
{ "question": "Hoạt chất có tên Latin là Acenocoumarolum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acenocoumarolum là ACENOCOUMAROL, có công thức hóa học C19H15NO6." },
{ "question": "Hoạt chất có công thức hóa học C19H15NO6 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C19H15NO6 là VIÊN NÉN ACENOCOUMAROL, có tên Latin là Tabellae Acenocoumaroli." },
{ "question": "Hoạt chất có tên Latin là Tabellae Acenocoumaroli có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Acenocoumaroli là VIÊN NÉN ACENOCOUMAROL, có công thức hóa học C19H15NO6." },
{ "question": "Hoạt chất có công thức hóa học C4H6N4O3S2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C4H6N4O3S2 là ACETAZOLAMID, có tên Latin là Acetazolamidum." },
{ "question": "Hoạt chất có tên Latin là Acetazolamidum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acetazolamidum là ACETAZOLAMID, có công thức hóa học C4H6N4O3S2." },
{ "question": "Hoạt chất có công thức hóa học C4H6N4O3S2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C4H6N4O3S2 là VIÊN NÉN ACETAZOLAMID, có tên Latin là Tabulettae Acetazolamidi." },
{ "question": "Hoạt chất có tên Latin là Tabulettae Acetazolamidi có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabulettae Acetazolamidi là VIÊN NÉN ACETAZOLAMID, có công thức hóa học C4H6N4O3S2." },
{ "question": "Hoạt chất có công thức hóa học C5H9NO3S có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C5H9NO3S là ACETYLCYSTEIN, có tên Latin là Acetylcystein." },
{ "question": "Hoạt chất có tên Latin là Acetylcystein có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acetylcystein là ACETYLCYSTEIN, có công thức hóa học C5H9NO3S." },
{ "question": "Hoạt chất có công thức hóa học C5H9NO3S có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C5H9NO3S là BỘT PHA HỖN DỊCH ACETYLCYSTEIN, có tên Latin là Pulveres Acetylcysteini." },
{ "question": "Hoạt chất có tên Latin là Pulveres Acetylcysteini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Pulveres Acetylcysteini là BỘT PHA HỖN DỊCH ACETYLCYSTEIN, có công thức hóa học C5H9NO3S." },
{ "question": "Hoạt chất có công thức hóa học C5H9NO3S có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C5H9NO3S là NANG ACETYLCYSTEIN, có tên Latin là Capsulae Acetylcysteini." },
{ "question": "Hoạt chất có tên Latin là Capsulae Acetylcysteini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Capsulae Acetylcysteini là NANG ACETYLCYSTEIN, có công thức hóa học C5H9NO3S." },
{ "question": "Hoạt chất có công thức hóa học C8H11N5O3 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C8H11N5O3 là ACICLOVIR, có tên Latin là Aciclovirum." },
{ "question": "Hoạt chất có tên Latin là Aciclovirum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Aciclovirum là ACICLOVIR, có công thức hóa học C8H11N5O3." },
{ "question": "Hoạt chất có công thức hóa học C8H11N5O3 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C8H11N5O3 là KEM ACICLOVIR, có tên Latin là Cremoris Acicloviri." },
{ "question": "Hoạt chất có tên Latin là Cremoris Acicloviri có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Cremoris Acicloviri là KEM ACICLOVIR, có công thức hóa học C8H11N5O3." },
{ "question": "Hoạt chất có công thức hóa học C8H11N5O3 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C8H11N5O3 là VIÊN NÉN ACICLOVIR, có tên Latin là Tabellae Acicloviri." },
{ "question": "Hoạt chất có tên Latin là Tabellae Acicloviri có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Acicloviri là VIÊN NÉN ACICLOVIR, có công thức hóa học C8H11N5O3." },
{ "question": "Hoạt chất có công thức hóa học C9H8O4 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C9H8O4 là ACID ACETYLSALICYLIC, có tên Latin là Acidum acetylsalicylicum." },
{ "question": "Hoạt chất có tên Latin là Acidum acetylsalicylicum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum acetylsalicylicum là ACID ACETYLSALICYLIC, có công thức hóa học C9H8O4." },
{ "question": "Hoạt chất có công thức hóa học C9H8O4 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C9H8O4 là VIÊN NÉN ACID ACETYLSALICYLIC, có tên Latin là Tabellae Acidi acetylsalicylici." },
{ "question": "Hoạt chất có tên Latin là Tabellae Acidi acetylsalicylici có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Acidi acetylsalicylici là VIÊN NÉN ACID ACETYLSALICYLIC, có công thức hóa học C9H8O4." },
{ "question": "Hoạt chất có công thức hóa học C9H8O4 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C9H8O4 là VIÊN NÉN BAO TAN TRONG RUỘT ACID, có tên Latin là Tabellae Acidi acetylsalicylici." },
{ "question": "Hoạt chất có tên Latin là Tabellae Acidi acetylsalicylici có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Acidi acetylsalicylici là VIÊN NÉN BAO TAN TRONG RUỘT ACID, có công thức hóa học C9H8O4." },
{ "question": "Hoạt chất có công thức hóa học C9H8O4 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C9H8O4 là VIÊN NÉN ASPIRIN VÀ CAFEIN, có tên Latin là Tabellae Aspirini et Coffeini." },
{ "question": "Hoạt chất có tên Latin là Tabellae Aspirini et Coffeini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Aspirini et Coffeini là VIÊN NÉN ASPIRIN VÀ CAFEIN, có công thức hóa học C9H8O4." },
{ "question": "Hoạt chất có công thức hóa học C6H13NO2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C6H13NO2 là ACID AMINOCAPROIC, có tên Latin là Acidum Aminocaproicum." },
{ "question": "Hoạt chất có tên Latin là Acidum Aminocaproicum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum Aminocaproicum là ACID AMINOCAPROIC, có công thức hóa học C6H13NO2." },
{ "question": "Hoạt chất có công thức hóa học C6H8O6 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C6H8O6 là ACID ASCORBIC, có tên Latin là Acidum Ascorbicum." },
{ "question": "Hoạt chất có tên Latin là Acidum Ascorbicum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum Ascorbicum là ACID ASCORBIC, có công thức hóa học C6H8O6." },
{ "question": "Hoạt chất có công thức hóa học C6H8O6 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C6H8O6 là THUỐC TIÊM ACID ASCORBIC, có tên Latin là Injectio Acidi ascorbici." },
{ "question": "Hoạt chất có tên Latin là Injectio Acidi ascorbici có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Injectio Acidi ascorbici là THUỐC TIÊM ACID ASCORBIC, có công thức hóa học C6H8O6." },
{ "question": "Hoạt chất có công thức hóa học C6H8O6 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C6H8O6 là VIÊN NÉN ACID ASCORBIC, có tên Latin là Tabellae Acidi ascorbici." },
{ "question": "Hoạt chất có tên Latin là Tabellae Acidi ascorbici có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Acidi ascorbici là VIÊN NÉN ACID ASCORBIC, có công thức hóa học C6H8O6." },
{ "question": "Hoạt chất có công thức hóa học C7H6O2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C7H6O2 là ACID BENZOIC, có tên Latin là Acidum benzoicum." },
{ "question": "Hoạt chất có tên Latin là Acidum benzoicum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum benzoicum là ACID BENZOIC, có công thức hóa học C7H6O2." },
{ "question": "Hoạt chất có công thức hóa học C7H6O2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C7H6O2 là THUỐC MỠ BENZOSALICYLIC, có tên Latin là Unguentum Benzosalicylici." },
{ "question": "Hoạt chất có tên Latin là Unguentum Benzosalicylici có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Unguentum Benzosalicylici là THUỐC MỠ BENZOSALICYLIC, có công thức hóa học C7H6O2." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là ACID BORIC, có tên Latin là Acidum boricum." },
{ "question": "Hoạt chất có tên Latin là Acidum boricum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum boricum là ACID BORIC, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là DUNG DỊCH ACID BORIC 3 %, có tên Latin là Solutio Acidi borici 3 %." },
{ "question": "Hoạt chất có tên Latin là Solutio Acidi borici 3 % có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Solutio Acidi borici 3 % là DUNG DỊCH ACID BORIC 3 %, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là THUỐC MỠ ACID BORIC 10 %, có tên Latin là Unguentum Acidi borici 10 %." },
{ "question": "Hoạt chất có tên Latin là Unguentum Acidi borici 10 % có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Unguentum Acidi borici 10 % là THUỐC MỠ ACID BORIC 10 %, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học C6H8O7 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C6H8O7 là ACID CITRIC NGẬM MỘT PHÂN TỬ NƯỚC, có tên Latin là Acidum citricum monohydricum." },
{ "question": "Hoạt chất có tên Latin là Acidum citricum monohydricum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum citricum monohydricum là ACID CITRIC NGẬM MỘT PHÂN TỬ NƯỚC, có công thức hóa học C6H8O7." },
{ "question": "Hoạt chất có công thức hóa học C19H19N7O6 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C19H19N7O6 là ACID FOLIC, có tên Latin là Acidum folicum." },
{ "question": "Hoạt chất có tên Latin là Acidum folicum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum folicum là ACID FOLIC, có công thức hóa học C19H19N7O6." },
{ "question": "Hoạt chất có công thức hóa học C19H19N7O6 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C19H19N7O6 là VIÊN NÉN ACID FOLIC, có tên Latin là Tabellae Acidi folici." },
{ "question": "Hoạt chất có tên Latin là Tabellae Acidi folici có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Acidi folici là VIÊN NÉN ACID FOLIC, có công thức hóa học C19H19N7O6." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là ACID HYDROCLORIC, có tên Latin là Acidum hydrochloricum." },
{ "question": "Hoạt chất có tên Latin là Acidum hydrochloricum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum hydrochloricum là ACID HYDROCLORIC, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là ACID HYDROCLORIC LOÃNG, có tên Latin là Acidum hydrochloricum dilutum." },
{ "question": "Hoạt chất có tên Latin là Acidum hydrochloricum dilutum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum hydrochloricum dilutum là ACID HYDROCLORIC LOÃNG, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học C15H15NO2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C15H15NO2 là ACID MEFENAMIC, có tên Latin là Acid mefenamic." },
{ "question": "Hoạt chất có tên Latin là Acid mefenamic có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acid mefenamic là ACID MEFENAMIC, có công thức hóa học C15H15NO2." },
{ "question": "Hoạt chất có công thức hóa học C15H15NO2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C15H15NO2 là VIÊN NÉN ACID MEFENAMIC, có tên Latin là Tabellae Acidi mefenamici." },
{ "question": "Hoạt chất có tên Latin là Tabellae Acidi mefenamici có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Acidi mefenamici là VIÊN NÉN ACID MEFENAMIC, có công thức hóa học C15H15NO2." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là ACID METHACRYLIC VÀ METHYL, có tên Latin là Acidi methacrylici et methylis methacrylatis polymerisatum (1 : 2)." },
{ "question": "Hoạt chất có tên Latin là Acidi methacrylici et methylis methacrylatis polymerisatum (1 : 2) có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidi methacrylici et methylis methacrylatis polymerisatum (1 : 2) là ACID METHACRYLIC VÀ METHYL, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học C12H12N2O3 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C12H12N2O3 là ACID NALIDIXIC, có tên Latin là Acid nalidixic." },
{ "question": "Hoạt chất có tên Latin là Acid nalidixic có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acid nalidixic là ACID NALIDIXIC, có công thức hóa học C12H12N2O3." },
{ "question": "Hoạt chất có công thức hóa học C12H12N2O3 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C12H12N2O3 là VIÊN NÉN ACID NALIDIXIC, có tên Latin là Tabellae Acidi nalidixici." },
{ "question": "Hoạt chất có tên Latin là Tabellae Acidi nalidixici có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Acidi nalidixici là VIÊN NÉN ACID NALIDIXIC, có công thức hóa học C12H12N2O3." },
{ "question": "Hoạt chất có công thức hóa học C6H5NO2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C6H5NO2 là ACID NICOTINIC, có tên Latin là Acidum nicotinicum." },
{ "question": "Hoạt chất có tên Latin là Acidum nicotinicum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum nicotinicum là ACID NICOTINIC, có công thức hóa học C6H5NO2." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là ACID METHACRYLIC VÀ ETHYL ACRYLAT, có tên Latin là Acidi methacrylici et ethylis acrylatis polymerisatint (1 : 1)." },
{ "question": "Hoạt chất có tên Latin là Acidi methacrylici et ethylis acrylatis polymerisatint (1 : 1) có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidi methacrylici et ethylis acrylatis polymerisatint (1 : 1) là ACID METHACRYLIC VÀ ETHYL ACRYLAT, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là DỊCH PHÂN TÁN 30 % CỦA ACID METHACRYLIC, có tên Latin là Acidi methacrylici et ethyls acrylatis polymerisati 1 : 1 dispersio 30 per centum." },
{ "question": "Hoạt chất có tên Latin là Acidi methacrylici et ethyls acrylatis polymerisati 1 : 1 dispersio 30 per centum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidi methacrylici et ethyls acrylatis polymerisati 1 : 1 dispersio 30 per centum là DỊCH PHÂN TÁN 30 % CỦA ACID METHACRYLIC, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là ACID METHACRYLIC VÀ METHYL, có tên Latin là Acidi methacrylici et methylis methacrylatis polymerisatum (I : I)." },
{ "question": "Hoạt chất có tên Latin là Acidi methacrylici et methylis methacrylatis polymerisatum (I : I) có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidi methacrylici et methylis methacrylatis polymerisatum (I : I) là ACID METHACRYLIC VÀ METHYL, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học C7H6O3 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C7H6O3 là ACID SALICYLIC, có tên Latin là Acidum salicylicium." },
{ "question": "Hoạt chất có tên Latin là Acidum salicylicium có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum salicylicium là ACID SALICYLIC, có công thức hóa học C7H6O3." },
{ "question": "Hoạt chất có công thức hóa học C8H15NO2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C8H15NO2 là ACID TRANEXAMIC, có tên Latin là Acidum tranexamicum." },
{ "question": "Hoạt chất có tên Latin là Acidum tranexamicum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Acidum tranexamicum là ACID TRANEXAMIC, có công thức hóa học C8H15NO2." },
{ "question": "Hoạt chất có công thức hóa học C8H15NO2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C8H15NO2 là NANG ACID TRANEXAMIC, có tên Latin là Capsulae Acidi tranexamici." },
{ "question": "Hoạt chất có tên Latin là Capsulae Acidi tranexamici có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Capsulae Acidi tranexamici là NANG ACID TRANEXAMIC, có công thức hóa học C8H15NO2." },
{ "question": "Hoạt chất có công thức hóa học C8H15NO2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C8H15NO2 là VIÊN NÉN ACID TRANEXAMIC, có tên Latin là Tabellae Acidi tranexamici." },
{ "question": "Hoạt chất có tên Latin là Tabellae Acidi tranexamici có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Acidi tranexamici là VIÊN NÉN ACID TRANEXAMIC, có công thức hóa học C8H15NO2." },
{ "question": "Hoạt chất có công thức hóa học C9H13NO3 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C9H13NO3 là ADRENALIN, có tên Latin là Adrenalinum." },
{ "question": "Hoạt chất có tên Latin là Adrenalinum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Adrenalinum là ADRENALIN, có công thức hóa học C9H13NO3." },
{ "question": "Hoạt chất có công thức hóa học C9H13NO3.C4H6O6 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C9H13NO3.C4H6O6 là ADRENALIN ACID TARTRAT, có tên Latin là Adrenalinum Acidum Tartratis." },
{ "question": "Hoạt chất có tên Latin là Adrenalinum Acidum Tartratis có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Adrenalinum Acidum Tartratis là ADRENALIN ACID TARTRAT, có công thức hóa học C9H13NO3.C4H6O6." },
{ "question": "Hoạt chất có công thức hóa học C9H13NO3 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C9H13NO3 là THUỐC TIÊM ADRENALIN, có tên Latin là Injectio Adrenalini." },
{ "question": "Hoạt chất có tên Latin là Injectio Adrenalini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Injectio Adrenalini là THUỐC TIÊM ADRENALIN, có công thức hóa học C9H13NO3." },
{ "question": "Hoạt chất có công thức hóa học C12H15N3O2S có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C12H15N3O2S là ALBENDAZOL, có tên Latin là Albendazolum." },
{ "question": "Hoạt chất có tên Latin là Albendazolum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Albendazolum là ALBENDAZOL, có công thức hóa học C12H15N3O2S." },
{ "question": "Hoạt chất có công thức hóa học C12H15N3O2S có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C12H15N3O2S là VIÊN NÉN ALBENDAZOL, có tên Latin là Tabellae Albendazoli." },
{ "question": "Hoạt chất có tên Latin là Tabellae Albendazoli có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Albendazoli là VIÊN NÉN ALBENDAZOL, có công thức hóa học C12H15N3O2S." },
{ "question": "Hoạt chất có công thức hóa học C18H22N2S.C4H6O6 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C18H22N2S.C4H6O6 là ALIMEMAZIN TARTRAT, có tên Latin là Alimemazini tartras." },
{ "question": "Hoạt chất có tên Latin là Alimemazini tartras có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Alimemazini tartras là ALIMEMAZIN TARTRAT, có công thức hóa học C18H22N2S.C4H6O6." },
{ "question": "Hoạt chất có công thức hóa học C18H22N2S.C4H6O6 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C18H22N2S.C4H6O6 là VIÊN NÉN ALIMEMAZIN, có tên Latin là Tabellae Alimemazini." },
{ "question": "Hoạt chất có tên Latin là Tabellae Alimemazini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Alimemazini là VIÊN NÉN ALIMEMAZIN, có công thức hóa học C18H22N2S.C4H6O6." },
{ "question": "Hoạt chất có công thức hóa học C5H4N4O có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C5H4N4O là ALOPURINOL, có tên Latin là Allopurinolum." },
{ "question": "Hoạt chất có tên Latin là Allopurinolum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Allopurinolum là ALOPURINOL, có công thức hóa học C5H4N4O." },
{ "question": "Hoạt chất có công thức hóa học C5H4N4O có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C5H4N4O là VIÊN NÉN ALLOPURINOL, có tên Latin là Tabellae Allopurinoli." },
{ "question": "Hoạt chất có tên Latin là Tabellae Allopurinoli có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Allopurinoli là VIÊN NÉN ALLOPURINOL, có công thức hóa học C5H4N4O." },
{ "question": "Hoạt chất có công thức hóa học C20H25N.C6H8O7 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C20H25N.C6H8O7 là ALVERIN CITRAT, có tên Latin là Alverini citras." },
{ "question": "Hoạt chất có tên Latin là Alverini citras có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Alverini citras là ALVERIN CITRAT, có công thức hóa học C20H25N.C6H8O7." },
{ "question": "Hoạt chất có công thức hóa học C20H25N.C6H8O7 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C20H25N.C6H8O7 là NANG ALVERIN, có tên Latin là Capsulae Alverini." },
{ "question": "Hoạt chất có tên Latin là Capsulae Alverini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Capsulae Alverini là NANG ALVERIN, có công thức hóa học C20H25N.C6H8O7." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là AMBROXOL HYDROCLORID, có tên Latin là Ambroxoli hydrochloridum." },
{ "question": "Hoạt chất có tên Latin là Ambroxoli hydrochloridum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Ambroxoli hydrochloridum là AMBROXOL HYDROCLORID, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là NANG AMBROXOL HYDROCLORID, có tên Latin là Capsulae Ambroxoli hydrochloridi." },
{ "question": "Hoạt chất có tên Latin là Capsulae Ambroxoli hydrochloridi có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Capsulae Ambroxoli hydrochloridi là NANG AMBROXOL HYDROCLORID, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là VIÊN NÉN AMBROXOL HYDROCLORID, có tên Latin là Tabellae Ambroxoli hydrochloridi." },
{ "question": "Hoạt chất có tên Latin là Tabellae Ambroxoli hydrochloridi có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Ambroxoli hydrochloridi là VIÊN NÉN AMBROXOL HYDROCLORID, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học C22H43N5O13 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C22H43N5O13 là AMIKACIN, có tên Latin là Amikacinum." },
{ "question": "Hoạt chất có tên Latin là Amikacinum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Amikacinum là AMIKACIN, có công thức hóa học C22H43N5O13." },
{ "question": "Hoạt chất có công thức hóa học C22H43N5O13 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C22H43N5O13 là THUỐC TIÊM AMIKACIN, có tên Latin là Injectio Amikacini." },
{ "question": "Hoạt chất có tên Latin là Injectio Amikacini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Injectio Amikacini là THUỐC TIÊM AMIKACIN, có công thức hóa học C22H43N5O13." },
{ "question": "Hoạt chất có công thức hóa học C7H8N4O2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C7H8N4O2 là AMINOPHYLLIN, có tên Latin là Aminophyllinum Theophyllinum ethylenediaminum." },
{ "question": "Hoạt chất có tên Latin là Aminophyllinum Theophyllinum ethylenediaminum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Aminophyllinum Theophyllinum ethylenediaminum là AMINOPHYLLIN, có công thức hóa học C7H8N4O2." },
{ "question": "Hoạt chất có công thức hóa học C2H8N2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C2H8N2 là THUỐC TIÊM AMINOPHYLIN, có tên Latin là Injectio Aminophyllini." },
{ "question": "Hoạt chất có tên Latin là Injectio Aminophyllini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Injectio Aminophyllini là THUỐC TIÊM AMINOPHYLIN, có công thức hóa học C2H8N2." },
{ "question": "Hoạt chất có công thức hóa học C7H8N4O2 có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C7H8N4O2 là VIÊN NÉN AMINOPHYLIN, có tên Latin là Tabellae Aminophyllini." },
{ "question": "Hoạt chất có tên Latin là Tabellae Aminophyllini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Aminophyllini là VIÊN NÉN AMINOPHYLIN, có công thức hóa học C7H8N4O2." },
{ "question": "Hoạt chất có công thức hóa học C25H29I2NO3. có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C25H29I2NO3. là AMIODARON HYDROCLORID, có tên Latin là Amiodaroni hydrochloridum." },
{ "question": "Hoạt chất có tên Latin là Amiodaroni hydrochloridum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Amiodaroni hydrochloridum là AMIODARON HYDROCLORID, có công thức hóa học C25H29I2NO3.." },
{ "question": "Hoạt chất có công thức hóa học C25H29I2NO3. có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C25H29I2NO3. là VIÊN NÉN AMIODARON, có tên Latin là Tabellae Amiodaroni." },
{ "question": "Hoạt chất có tên Latin là Tabellae Amiodaroni có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Amiodaroni là VIÊN NÉN AMIODARON, có công thức hóa học C25H29I2NO3.." },
{ "question": "Hoạt chất có công thức hóa học C20H23N. có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C20H23N. là AMITRIPTYLIN HYDROCLORID, có tên Latin là Amitriptylini hydrochloridum." },
{ "question": "Hoạt chất có tên Latin là Amitriptylini hydrochloridum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Amitriptylini hydrochloridum là AMITRIPTYLIN HYDROCLORID, có công thức hóa học C20H23N.." },
{ "question": "Hoạt chất có công thức hóa học C20H23N. có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C20H23N. là VIÊN NÉN AMITRIPTYLIN, có tên Latin là Tabellae Amitriptylini." },
{ "question": "Hoạt chất có tên Latin là Tabellae Amitriptylini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Amitriptylini là VIÊN NÉN AMITRIPTYLIN, có công thức hóa học C20H23N.." },
{ "question": "Hoạt chất có công thức hóa học C6H6O3S có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C6H6O3S là AMLODIPIN BESILAT, có tên Latin là Amlodipini besilas." },
{ "question": "Hoạt chất có tên Latin là Amlodipini besilas có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Amlodipini besilas là AMLODIPIN BESILAT, có công thức hóa học C6H6O3S." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là VIÊN NÉN AMLODIPIN, có tên Latin là Tabellae Amlodipini." },
{ "question": "Hoạt chất có tên Latin là Tabellae Amlodipini có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Amlodipini là VIÊN NÉN AMLODIPIN, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là AMODIAQUIN HYDROCLORID, có tên Latin là Amodiaquini hydrochloridum." },
{ "question": "Hoạt chất có tên Latin là Amodiaquini hydrochloridum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Amodiaquini hydrochloridum là AMODIAQUIN HYDROCLORID, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là VIÊN NÉN AMODIAQUIN HYDROCLORID, có tên Latin là Tabellae Amodiaquini hydrochloridi." },
{ "question": "Hoạt chất có tên Latin là Tabellae Amodiaquini hydrochloridi có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Tabellae Amodiaquini hydrochloridi là VIÊN NÉN AMODIAQUIN HYDROCLORID, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là AMONI CLORID, có tên Latin là Ammonii chloridum." },
{ "question": "Hoạt chất có tên Latin là Ammonii chloridum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Ammonii chloridum là AMONI CLORID, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học không có thông tin có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học không có thông tin là AMOXICILIN NATRI, có tên Latin là Amoxicillin natricum." },
{ "question": "Hoạt chất có tên Latin là Amoxicillin natricum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Amoxicillin natricum là AMOXICILIN NATRI, có công thức hóa học không có thông tin." },
{ "question": "Hoạt chất có công thức hóa học C16H19N3O5S có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C16H19N3O5S là BỘT PHA TIÊM AMOXICILIN, có tên Latin là Amoxicillini pro Injectione." },
{ "question": "Hoạt chất có tên Latin là Amoxicillini pro Injectione có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Amoxicillini pro Injectione là BỘT PHA TIÊM AMOXICILIN, có công thức hóa học C16H19N3O5S." },
{ "question": "Hoạt chất có công thức hóa học C16H19N3O5S có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C16H19N3O5S là BỘT PHA TIÊM AMOXICILIN VÀ ACID CLAVULANIC, có tên Latin là Amoxicillini et Acidi clavulanici pulvis pro injectione." },
{ "question": "Hoạt chất có tên Latin là Amoxicillini et Acidi clavulanici pulvis pro injectione có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Amoxicillini et Acidi clavulanici pulvis pro injectione là BỘT PHA TIÊM AMOXICILIN VÀ ACID CLAVULANIC, có công thức hóa học C16H19N3O5S." },
{ "question": "Hoạt chất có công thức hóa học C16H19N3O5S có tên Latin là gì?", "answer": "Hoạt chất có công thức hóa học C16H19N3O5S là AMOXICILIN TRIHYDRAT, có tên Latin là Amoxicillinum trihydricum." },
{ "question": "Hoạt chất có tên Latin là Amoxicillinum trihydricum có công thức hóa học là gì?", "answer": "Hoạt chất có tên Latin là Amoxicillinum trihydricum là AMOXICILIN TRIHYDRAT, có công thức hóa học C16H19N3O5S." },
]


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
        
        time.sleep(2) # Delay nhẹ tránh rate limit

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
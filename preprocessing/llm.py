import os
import google.generativeai as genai
from dotenv import load_dotenv

# ========================================================
# 1. TỰ ĐỘNG CẤU HÌNH (Auto-Config)
# ========================================================

# Tự động tìm file key.env ở thư mục gốc (lùi ra 2 cấp từ preprocessing/kgraph)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '..', '..', 'key.env')

# Nếu không thấy, thử tìm ở cấp cha gần nhất (lùi 1 cấp)
if not os.path.exists(env_path):
    env_path = os.path.join(current_dir, '..', 'key.env')

# Đọc file .env
load_dotenv(env_path)

# Lấy API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    # Key dự phòng (Fallback) nếu file .env bị lỗi hoặc chưa tạo
    print("⚠️ Cảnh báo: Không đọc được key.env")

# Cấu hình Gemini
genai.configure(api_key=GOOGLE_API_KEY.strip())

# --- THAY ĐỔI THEO YÊU CẦU: Dùng Model 2.0 Flash ---
MODEL_NAME = "models/gemini-2.0-flash"

# ========================================================
# 2. CẤU HÌNH THAM SỐ (Generation Config)
# ========================================================
generation_config = {
  "temperature": 0,       # Nhiệt độ = 0 để trả lời chính xác, không bịa
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 50000,
}

safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

# Khởi tạo Model
try:
    model = genai.GenerativeModel(model_name=MODEL_NAME,
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
except Exception as e:
    print(f"❌ Lỗi khởi tạo Model {MODEL_NAME}: {e}")

# ========================================================
# 3. CÁC HÀM GIAO TIẾP (API Wrappers - Giữ nguyên tên hàm gốc)
# ========================================================

def get_GPT(text):
    """
    Hàm này tên là GPT (để khớp với code cũ của tác giả),
    nhưng thực tế sẽ gọi Gemini để bạn không mất tiền OpenAI.
    """
    return get_gemini(text)
 
def get_gemini(text): 
    """
    Hàm gọi Gemini chính.
    """
    try:
        # Gọi API sinh nội dung
        response = model.generate_content([text])
        return response.text
    except Exception as e:
        # Xử lý lỗi nếu Google chặn hoặc hết quota
        err_msg = str(e)
        if "429" in err_msg or "Quota" in err_msg:
            return "Lỗi: Hết Quota (Limit Exceeded). Vui lòng thử lại sau."
        return f"Lỗi Gemini: {err_msg}"

# ========================================================
# 4. CHẠY TEST NHANH
# ========================================================
if __name__ == "__main__":
    print(f"--- Đang test llm.py ---")
    print(f"✅ Model đang dùng: {MODEL_NAME}")
    print(f"🔑 Key đang dùng: ...{GOOGLE_API_KEY[-5:]}")
    
    while True:
        q = input("\nBạn hỏi (gõ 'exit' để thoát): ")
        if q.lower() in ['exit', 'quit']: break
        
        print("Bot đáp:", get_gemini(q))
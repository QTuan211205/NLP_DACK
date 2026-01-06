import os
import json
import time
import nltk
import warnings
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Thư viện tính toán điểm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score

# Thư viện AI
from langchain_google_genai import ChatGoogleGenerativeAI

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# Load môi trường
load_dotenv("key.env")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Khởi tạo Model Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Hoặc gemini-pro tùy tài khoản của bạn
    temperature=0, # Giữ temperature thấp để đánh giá tính chính xác
    google_api_key=google_api_key
)

def get_gemini(text):
    # Trích xuất nội dung từ AIMessage object
    response = llm.invoke([text])
    return response.content

def call_model_with_retry(model_func, prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_func(prompt)
        except Exception as e:
            print(f"Lỗi: {e}. Đang thử lại lần {attempt+1}...")
            time.sleep(2)
    return ""

# Tải tài nguyên NLTK
nltk.download("wordnet")
nltk.download("punkt")

# Khởi tạo ROUGE
rouge = Rouge()
smoothing_function = SmoothingFunction().method1

def get_scores(hypothesis, reference):
    if not hypothesis or not reference:
        return 0, 0, 0
    
    # Tokenize cho BLEU và METEOR
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    reference_tokens = nltk.word_tokenize(reference.lower())
    
    # BLEU Score
    bleu = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing_function)
    
    # ROUGE Score (Sử dụng chuỗi văn bản gốc)
    try:
        rouge_scores = rouge.get_scores(hypothesis.lower(), reference.lower())
        rouge_score = rouge_scores[0]["rouge-l"]["f"]
    except:
        rouge_score = 0
        
    # METEOR Score
    meteor = meteor_score([reference_tokens], hypothesis_tokens)
    
    return bleu, rouge_score, meteor

# Đường dẫn file và thư mục
results_dir = "results"
logs_dir = "logs"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# ============================
# CẤU HÌNH DATASET
# ============================

DATASETS = {
    "1-hop": "data/benchmark/1hop.json",
    "2-hop": "data/benchmark/2hop.json",
}

test_limit = 200

# ============================
# HÀM CHẠY EVALUATION
# ============================

def run_zero_shot(dataset_name, file_path):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"\n🚀 Bắt đầu Zero-shot {dataset_name} ({min(test_limit, len(data))} câu hỏi)")

    scores = {"BLEU": [], "ROUGE": [], "METEOR": []}
    logs = []
    inference_times = []

    for x in tqdm(data[:test_limit], desc=f"{dataset_name}"):

        PROMPT = f"""
        Bạn là một dược sĩ lâm sàng và chuyên gia về Dược điển Việt Nam. 
        Hãy trả lời câu hỏi sau một cách chính xác, ngắn gọn và dựa trên kiến thức chuyên môn y dược.

        - Trả lời thẳng vào vấn đề.
        - Giữ độ chính xác cao về tên thuốc và công thức hóa học.
        
        Câu hỏi: {x["question"]}
        """

        start_time = time.time()
        gemini_result = call_model_with_retry(get_gemini, PROMPT)
        end_time = time.time()

        inference_times.append(end_time - start_time)

        reference = x["answer"]

        bleu, rouge_val, meteor = get_scores(gemini_result, reference)

        scores["BLEU"].append(bleu)
        scores["ROUGE"].append(rouge_val)
        scores["METEOR"].append(meteor)

        logs.append({
            "hop_type": dataset_name,
            "question": x["question"],
            "ground_truth": reference,
            "model_answer": gemini_result,
            "BLEU": bleu,
            "ROUGE": rouge_val,
            "METEOR": meteor,
            "time": end_time - start_time
        })

    # ============================
    # GHI KẾT QUẢ
    # ============================

    avg_time = sum(inference_times) / len(inference_times)

    result_path = os.path.join(results_dir, f"gemini_zero_shot_{dataset_name}.txt")
    log_path = os.path.join(logs_dir, f"gemini_zero_shot_{dataset_name}.json")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"{dataset_name} Zero-shot Results\n")
        f.write(f"Average inference time: {avg_time:.2f} seconds\n\n")
        for metric, values in scores.items():
            f.write(f"{metric}: {sum(values)/len(values):.4f}\n")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    print(f"✅ Hoàn thành {dataset_name} | Avg time: {avg_time:.2f}s")
    print(f"📄 Results: {result_path}")
    print(f"🧾 Logs: {log_path}")

# ============================
# CHẠY CẢ 1-HOP & 2-HOP
# ============================

for hop_name, path in DATASETS.items():
    run_zero_shot(hop_name, path)

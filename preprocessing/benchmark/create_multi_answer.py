from utils import read_json, save_json
import re
from collections import defaultdict
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
import os

warnings.filterwarnings("ignore")

class QuestionProcessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.grouped_data = defaultdict(list)
        self.merged_data = []

    def extract_content(self, question):
        """Trích xuất nội dung trong ngoặc [] để so sánh độ tương đồng."""
        try: 
            match = re.search(r'\[(.*?)\]', question)
            return match.group(1).lower() if match else None
        except:
            return None

    def jaccard_similarity(self, set1, set2):
        """Tính độ tương đồng Jaccard thủ công (Nhanh và ổn định hơn MLB cho tập nhỏ)."""
        if not set1 or not set2: return 0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def remove_duplicates(self, list_of_dicts):
        """Loại bỏ các câu trùng lặp hoàn toàn, an toàn với dữ liệu thiếu key."""
        seen = set()
        unique_dicts = []
        for d in list_of_dicts:
            # Sử dụng .get() để lấy giá trị, nếu không có key sẽ trả về 'N/A'
            q = d.get('question', '')
            q_type = d.get('question_type', 'N/A') 
            ans = d.get('answer', '')
            
            # Tạo tuple để kiểm tra trùng lặp
            dict_tuple = (q, q_type, ans)
            
            if dict_tuple not in seen:
                seen.add(dict_tuple)
                unique_dicts.append(d)
        return unique_dicts

    def process_questions(self):
        data = read_json(self.input_file)
        if not data: 
            return
        
        data = self.remove_duplicates(data)

        # BƯỚC 1: Nhóm theo cặp (Thực thể + Quan hệ)
        # Thay vì chỉ dùng q_type, ta dùng thêm content_key và relation để phân loại
        for item in data:
            content = self.extract_content(item["question"])
            rel = item.get("relation", "unknown") # Lấy trường relation từ file triples
            
            if content is None: continue
            
            # Tạo một key định danh duy nhất cho nhóm câu hỏi cùng chủ đề và cùng loại thông tin
            # Ví dụ: "paracetamol_ten_latin"
            group_key = f"{content}_{rel}"
            self.grouped_data[group_key].append(item)

        # BƯỚC 2: Xử lý gộp trong từng nhóm đã phân loại đúng
        for group_key, items in self.grouped_data.items():
            # Trong cùng một group_key chắc chắn đã cùng thực thể và cùng loại quan hệ
            # Bây giờ chỉ cần gộp các câu trả lời khác nhau (nếu có) cho cùng 1 ý đó
            base_item = items[0]
            
            raw_answers = [str(it["answer"]).strip() for it in items]
            valid_answers = [a for a in raw_answers if a.lower() != "không có thông tin" and a != ""]
            
            # Loại bỏ trùng lặp trong nội dung câu trả lời
            unique_answers = sorted(list(set(valid_answers)))
            merged_answer = " | ".join(unique_answers) if unique_answers else "Không có thông tin"
            
            self.merged_data.append({
                "question": base_item["question"],
                "question_type": base_item.get("question_type", "default"),
                "relation": base_item.get("relation", "unknown"), # Giữ lại relation để kiểm tra
                "answer": merged_answer
            })

    def save_processed_questions(self):
        # ĐÃ SỬA: Bỏ self. để gọi đúng hàm từ utils.py
        save_json(self.merged_data, self.output_file)
        print(f"--- Đã lưu {len(self.merged_data)} câu hỏi vào: {self.output_file}")


def main(input_filename):
    if os.path.exists(input_filename):
        print(f"Đang xử lý gộp câu trả lời cho: {input_filename}")
        processor = QuestionProcessor(input_filename, input_filename)
        processor.process_questions()
        processor.save_processed_questions()
    else:
        print(f"Lỗi: Không tìm thấy file {input_filename}")

if __name__ == "__main__":
    # KIỂM TRA LẠI: Tên file của bạn có dấu gạch dưới hay không?
    file_1hop = '../../data/benchmark/1hop.json' 
    file_2hop = '../../data/benchmark/2hop.json'
    
    main(file_1hop)
    main(file_2hop)
    print("Hoàn tất quy trình hậu xử lý dữ liệu!")
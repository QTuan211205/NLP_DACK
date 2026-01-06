import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils import read_json, save_json
from preprocessing.llm import get_GPT
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json 

# Ignore all warnings
warnings.filterwarnings("ignore")

# Class 1: Từ tên Hoạt chất hỏi về các thuộc tính (Drug -> Property)
class Question_hoatchat_to_X:
    def __init__(self, input_filename, output_filename):
        self.input_filename = input_filename
        self.output_filename = output_filename
        # Cập nhật theo các cột trong CSV của bạn
        self.relation_dict = {
            "Ten_Latin": "tên_latin",
            "Cong_Thuc_Hoa_Hoc": "công_thức_hóa_học",
            "Mo_Ta_Chung": "mô_tả_chung",
            "Tinh_Chat": "tính_chất",
            "Dinh_Tinh": "định_tính",
            "Dinh_Luong": "định_lượng",
            "Bao_Quan": "bảo_quản",
            "Loai_Thuoc": "loại_thuốc",
            "Ham_Luong_Yeu_Cau": "hàm_lượng_yêu_cầu",
            "Tap_Chat_Va_Do_Tinh_Khiet": "tạp_chất_và_độ_tinh_khiết",
            "Do_Hoa_Tan": "độ_hòa_tan"
        }

    def process_data(self, data):
        for item in data:
            if item['relation'] in self.relation_dict:
                item['relation'] = self.relation_dict[item['relation']]
        return data

    def create_question(self, item):
        header = item['header']
        rel = item['relation']
        
        if rel == 'tên_latin':
            return f"Tên Latin của hoạt chất [{header}] là gì?"
        elif rel == 'công_thức_hóa_học':
            return f"Công thức hóa học của [{header}] được viết như thế nào?"
        elif rel == 'mô_tả_chung':
            return f"Mô tả chung về hoạt chất [{header}]?"
        elif rel == 'tính_chất':
            return f"Tính chất vật lý và hóa học của [{header}] như thế nào?"
        elif rel == 'định_tính':
            return f"Các phương pháp định tính của [{header}] là gì?"
        elif rel == 'định_lượng':
            return f"Cách tiến hành định lượng cho [{header}]?"
        elif rel == 'bảo_quản':
            return f"Yêu cầu bảo quản đối với hoạt chất [{header}] như thế nào?"
        elif rel == 'loại_thuốc':
            return f"Hoạt chất [{header}] thuộc nhóm hoặc loại thuốc nào?"
        elif rel == 'hàm_lượng_yêu_cầu':
            return f"Hàm lượng yêu cầu của chế phẩm [{header}] là bao nhiêu?"
        elif rel == 'tạp_chất_và_độ_tinh_khiết':
            return f"Tiêu chuẩn về tạp chất và độ tinh khiết của [{header}]?"
        elif rel == 'độ_hòa_tan':
            return f"Độ hòa tan của [{header}] trong các dung môi?"
        else:
            return None

    def generate_question(self, item):
        # Đổi ngữ cảnh từ Bác sĩ sang Dược sĩ/Chuyên gia kiểm nghiệm
        prompt = f"""Imagine you are a pharmacist or a drug quality control expert.
        Based on the provided technical question [{item['question']}], create a natural, human-like question that a pharmacy student or a professional might ask. 
        Return the question in JSON format, preserving the brackets [] around the entity.
        If the answer is "Không có thông tin", return {{"question": ""}}.
        Example: {{"question": "Bạn có thể cho biết công thức hóa học của hoạt chất [{item['header']}] không?"}}
        """
        result = get_GPT(prompt)
        try:
            result = eval(result[result.find('{'): result.rfind('}') + 1])
        except:
            return None
        if not result or result.get('question') == "":
            return None
        return result['question']

    def process_item(self, item):
        if item['answer'] == "Không có thông tin" or not item['answer']:
            return None
        raw_q = self.create_question(item)
        if not raw_q: return None
        item['question'] = raw_q
        generated_question = self.generate_question(item)
        if generated_question:
            item['question'] = generated_question
            return item
        return None

    def run_processing(self):
        json_data = read_json(self.input_filename)
        processed_data = self.process_data(json_data)
        save_data = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_item = {executor.submit(self.process_item, i): i for i in processed_data}
            for future in tqdm(as_completed(future_to_item), total=len(processed_data)):
                item = future.result()
                if item:
                    save_data.append(item)
        save_json(save_data, self.output_filename)

# Class 2: Từ thuộc tính hỏi ngược lại tên hoạt chất (Property -> Drug)
class Question_X_to_hoatchat:
    def __init__(self, input_filename, output_filename):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.relation_dict = {
            "Ten_Latin": "tên_latin",
            "Cong_Thuc_Hoa_Hoc": "công_thức_hóa_học",
            "Mo_Ta_Chung": "mô_tả_chung",
            "Tinh_Chat": "tính_chất",
            "Dinh_Tinh": "định_tính",
            "Dinh_Luong": "định_lượng",
            "Bao_Quan": "bảo_quản",
            "Loai_Thuoc": "loại_thuốc",
            "Ham_Luong_Yeu_Cau": "hàm_lượng_yêu_cầu",
            "Tap_Chat_Va_Do_Tinh_Khiet": "tạp_chất_và_độ_tinh_khiết",
            "Do_Hoa_Tan": "độ_hòa_tan"
        }

    def process_data(self, data):
        for item in data:
            if item['relation'] in self.relation_dict:
                item['relation'] = self.relation_dict[item['relation']]
        return data

    def create_question(self, item):
        tail = item['tail']
        rel = item['relation']
        
        if rel == 'tên_latin':
            return f"Hoạt chất nào có tên Latin là [{tail}]?"
        elif rel == 'công_thức_hóa_học':
            return f"Chất nào được xác định bởi công thức hóa học [{tail}]?"
        elif rel == 'loại_thuốc':
            return f"Kể tên một loại thuốc thuộc nhóm [{tail}]?"
        elif rel == 'bảo_quản':
            return f"Hoạt chất nào yêu cầu điều kiện bảo quản là [{tail}]?"
        elif rel == 'tính_chất':
            return f"Dựa vào tính chất [{tail}], đây là hoạt chất gì?"
        elif rel == 'độ_hòa_tan':
            return f"Chất nào có đặc tính hòa tan là [{tail}]?"
        else:
            return f"Thông tin [{tail}] thuộc về hoạt chất nào?"

    def generate_question(self, text):
        prompt = f"""Imagine you are a chemistry professor testing a student.
        Create a natural question based on the fact: [{text}].
        The answer to the question should be the name of a drug/chemical.
        Requirements:
        - Keep the bracket [] for the entity.
        - If the content in [] is too long, summarize it within the brackets.
        - Return JSON: {{"question": "..."}}.
        """
        result = get_GPT(prompt)
        try:
            result = eval(result[result.find('{'): result.rfind('}') + 1])
        except:
            return "NULL"
        return result.get('question', "NULL")

    def process_item(self, item):
        if item['answer'] == "Không có thông tin" or not item['tail']:
            return None
        raw_q = self.create_question(item)
        generated_question = self.generate_question(raw_q)
        if generated_question == "NULL":
            return None
        item['question'] = generated_question
        return item

    def run_processing(self):
        json_data = read_json(self.input_filename)
        processed_data = self.process_data(json_data)
        save_data = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_item = {executor.submit(self.process_item, i): i for i in processed_data}
            for future in tqdm(as_completed(future_to_item), total=len(processed_data)):
                item = future.result()
                if item:
                    save_data.append(item)
        save_json(save_data, self.output_filename)

def merge_json_files(file1, file2, output_file):
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    merged_data = data1 + data2
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    generator1 = Question_hoatchat_to_X(
        '../../data/benchmark/triples.json', 
        '../../data/benchmark/1hop_drug_to_X.json'
    )
    generator1.run_processing()

    generator2 = Question_X_to_hoatchat(
        '../../data/benchmark/triples.json', 
        '../../data/benchmark/1hop_X_to_drug.json'
    )
    generator2.run_processing()

    merge_json_files(
        '../../data/benchmark/1hop_drug_to_X.json', 
        '../../data/benchmark/1hop_X_to_drug.json', 
        '../../data/benchmark/1hop.json'
    )
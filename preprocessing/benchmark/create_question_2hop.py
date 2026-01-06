import sys
import os
import warnings
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Thêm đường dẫn thư mục gốc vào sys.path để Python tìm thấy package preprocessing và utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils import read_json, save_json
from preprocessing.llm import get_GPT

warnings.filterwarnings("ignore")

def process_data(data, relation_dict):
    for item in data:
        if item['relation'] in relation_dict:
            item['relation'] = relation_dict[item['relation']]
    return data

def create_question(i):
    rel1 = i['relation_1']
    rel2 = i['relation_2']
    t1 = i['tail_1']

    if rel1 == 'công_thức_hóa_học' and rel2 == 'bảo_quản':
        return f"Hoạt chất có công thức hóa học là [{t1}] yêu cầu điều kiện bảo quản như thế nào?"
    elif rel1 == 'tên_latin' and rel2 == 'loại_thuốc':
        return f"Thuốc có tên Latin [{t1}] thuộc nhóm dược lý nào?"
    elif rel1 == 'công_thức_hóa_học' and rel2 == 'định_lượng':
        return f"Phương pháp định lượng dành cho dược chất có công thức [{t1}] là gì?"
    elif rel1 == 'tên_latin' and rel2 == 'tính_chất':
        return f"Mô tả các tính chất vật lý của hoạt chất có tên Latin là [{t1}]?"
    elif rel1 == 'công_thức_hóa_học' and rel2 == 'loại_thuốc':
        return f"Dược chất mang công thức [{t1}] được phân vào loại thuốc nào?"
    elif rel1 == 'tên_latin' and rel2 == 'độ_hòa_tan':
        return f"Độ hòa tan của hoạt chất có tên Latin [{t1}] được quy định như thế nào?"
    elif rel1 == 'tính_chất' and rel2 == 'định_tính':
        return f"Với dược chất có tính chất [{t1}], quy trình định tính cụ thể là gì?"
    elif rel1 == 'mô_tả_chung' and rel2 == 'bảo_quản':
        return f"Dựa trên mô tả [{t1}], thuốc này cần được bảo quản ra sao?"
    elif rel2 == 'công_thức_hóa_học' and rel1 == 'loại_thuốc':
        return f"Loại thuốc [{t1}] thường có hoạt chất với công thức hóa học là gì?"
    return "NULL"

def get_prompt(text):
    # Đã chuyển toàn bộ chỉ thị sang tiếng Việt để AI trả lời tiếng Việt
    prompt = f"""Bạn là một chuyên gia về dược phẩm và kiểm nghiệm thuốc.
            Hãy tạo một câu hỏi tiếng Việt tự nhiên, chuyên sâu dựa trên bản thảo thô sau: [{text}].
            
            Trả về định dạng JSON: {{"question": ""}}.
            
            Yêu cầu bắt buộc:
            - Phải trả lời bằng TIẾNG VIỆT.
            - Giữ nguyên dấu ngoặc [] cho nội dung thực thể (entity).
            - Nếu nội dung trong [] quá dài hoặc là một danh sách, hãy tóm tắt lại thành vài ý chính.
            - Câu hỏi phải là một câu văn hoàn chỉnh, trôi chảy và mang tính chuyên môn.
            - Độ dài câu hỏi phải ít hơn 30 từ.
            - Nếu bản thảo thô không có thông tin cụ thể, trả về JSON rỗng {{}}.
            """
    result = get_GPT(prompt)
    try:
        # Trích xuất JSON từ phản hồi của AI
        result = eval(result[result.find('{'): result.rfind('}') + 1])
    except:
        return "NULL"
    
    if result == {} or not result.get('question'):
        return "NULL"
    return result['question']

def process_item(i):
    if i['answer'] == "Không có thông tin" or not i['answer']:
        return None
    i['question'] = get_prompt(i['question'])
    if i['question'] == "NULL":
        return None
    return i

def main(input_filename, output_filename):
    relation_dict = {
        "Ten_Latin": "tên_latin",
        "Cong_Thuc_Hoa_Hoc": "công_thức_hóa_học",
        "Mo_Ta_Chung": "mô_tả_chung",
        "Tinh_Chat": "tính_chất",
        "Dinh_Tinh": "định_tính",
        "Dinh_Luong": "định_lượng",
        "Bao_Quan": "bảo_quản",
        "Loai_Thuoc": "loại_thuốc",
        "Do_Hoa_Tan": "độ_hòa_tan"
    }

    json_data = read_json(input_filename)
    processed_data = process_data(json_data, relation_dict)

    grouped_data = defaultdict(list)
    for item in processed_data:
        grouped_data[item['header']].append(item)

    merged_data = []
    for header, items in grouped_data.items():
        if len(items) >= 2:
            for item1, item2 in combinations(items, 2):
                merged_data.append({
                    'header': header,
                    'relation_1': item1['relation'],
                    'tail_1': item1['tail'],
                    'relation_2': item2['relation'],
                    'tail_2': item2['tail']
                })
                merged_data.append({
                    'header': header,
                    'relation_1': item2['relation'],
                    'tail_1': item2['tail'],
                    'relation_2': item1['relation'],
                    'tail_2': item1['tail']
                })

    data_to_gpt = []
    for i in merged_data:
        ques = create_question(i)
        if ques == "NULL":
            continue
        data_to_gpt.append({
            "question": ques,
            "question_type": f"{i['relation_1']}_to_{i['relation_2']}",
            "answer": i['tail_2'],
        })

    save_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_item = {executor.submit(process_item, i): i for i in data_to_gpt}
        for future in tqdm(as_completed(future_to_item), total=len(data_to_gpt)):
            item = future.result()
            if item is not None:
                save_data.append(item)

    save_json(save_data, output_filename)

if __name__ == "__main__":
    input_file = '../../data/benchmark/triples.json'
    output_file = '../../data/benchmark/2hop.json'
    main(input_file, output_file)
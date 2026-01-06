import pandas as pd
import json
import numpy as np

def create_list_of_dicts(df):
    result = []
    for _, row in df.iterrows():
        # Đổi từ 'disease_name' sang 'Ten_Hoat_Chat'
        drug_name = str(row['Ten_Hoat_Chat']).strip()
        
        for col in df.columns:
            # Bỏ qua cột tên chính
            if col == 'Ten_Hoat_Chat':
                continue
                
            val = str(row[col]).strip()
            # Làm sạch dữ liệu chuỗi
            val = val.replace("[", "").replace("]", "").replace('\"',"").replace("\'","")
            
            # Kiểm tra dữ liệu hợp lệ (không rỗng, không phải nan)
            if val != "" and val.lower() != "nan" and val.lower() != "không có thông tin":
                result.append({
                    "header": drug_name,
                    "relation": col,
                    "tail": val,
                    "answer": val  # Thêm trường này để class tạo câu hỏi có thể đọc được
                })
    return result

# Đổi tên file CSV đầu vào của bạn tại đây
df = pd.read_csv("../../data/data_midterm.csv") 

# Tạo danh sách triples
list_of_dicts = create_list_of_dicts(df)

# Lưu vào đúng đường dẫn mà file create_question_1hop.py sẽ đọc
output_file = '../../data/benchmark/triples.json'

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(list_of_dicts, f, ensure_ascii=False, indent=4)

print(f"Đã tạo xong file triples tại: {output_file}")
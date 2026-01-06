import docx
import pandas as pd
import re
import os

# --- 1. CÁC HÀM XỬ LÝ TEXT ---

def normalize_chemistry_text(text):
    if not text: return ""
    subs_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    text = text.translate(subs_map)
    text = text.replace("½", "1/2").replace("⅓", "1/3").replace("¼", "1/4")
    text = text.replace('\xa0', ' ')
    return text

def clean_text(text):
    if not text: return ""
    text = normalize_chemistry_text(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_chemical_formula(text):
    if not text: return ""
    text = normalize_chemistry_text(text)
    match = re.search(r'\bC\d+H\d+[A-Z0-9\(\)\.]*(\.[\d\/]*H\d*[A-Z0-9]*)?(\.[\d\/]*[A-Z][a-z]?[A-Z0-9]*)?\b', text)
    if match:
        formula = match.group(0)
        if len(formula) > 3 and any(c.isdigit() for c in formula):
            return formula
    return ""

def is_image_line(text):
    text = text.strip()
    if text.startswith("[Image"): return True
    if text.startswith("(Hình"): return True
    if re.match(r'^hình\s*[\d\.]+', text.lower()): return True
    return False

# --- 2. HÀM ĐỌC DOCX ---

def parse_docx_to_df(docx_path):
    filename = os.path.basename(docx_path)
    print(f"--> Đang đọc file: {filename}")
    
    if not os.path.exists(docx_path):
        print(f"LỖI: Không tìm thấy file {docx_path}")
        return pd.DataFrame()

    doc = docx.Document(docx_path)
    full_text = "\n".join([p.text for p in doc.paragraphs])
    
    # Logic tách thuốc (Hybrid)
    if '</break>' in full_text:
        print("    [Info] Chế độ tách: Thẻ </break>")
        raw_drugs = full_text.split('</break>')
    else:
        print("    [Info] Chế độ tách: Regex số thứ tự")
        raw_drugs = re.split(r'\n(?=\d+\.\d+\.)', full_text)

    data = []
    
    # Các cột chính
    core_columns = [
        "Ten_Hoat_Chat", "Ten_Latin", "Cong_Thuc_Hoa_Hoc", "Mo_Ta_Chung",
        "Tinh_Chat", "Dinh_Tinh", "Dinh_Luong", "Bao_Quan", 
        "Loai_Thuoc", "Ham_Luong_Yeu_Cau", "Tap_Chat_Va_Do_Tinh_Khiet", 
        "Do_Hoa_Tan"
    ]
    
    # Mapping Header
    headers_routing = {
        "TÍNH CHẤT": "Tinh_Chat", "ĐỊNH TÍNH": "Dinh_Tinh", "ĐỊNH LƯỢNG": "Dinh_Luong",
        "BẢO QUẢN": "Bao_Quan", "LOẠI THUỐC": "Loai_Thuoc", "HÀM LƯỢNG": "Ham_Luong_Yeu_Cau",
        "TẠP CHẤT": "Tap_Chat_Va_Do_Tinh_Khiet", "ĐỘ HÒA TAN": "Do_Hoa_Tan",
        # Các chỉ tiêu phụ -> Gom vào Tạp chất
        "PH": "Tap_Chat_Va_Do_Tinh_Khiet", 
        "NƯỚC": "Tap_Chat_Va_Do_Tinh_Khiet",
        "MẤT KHỐI LƯỢNG": "Tap_Chat_Va_Do_Tinh_Khiet", "CẶN": "Tap_Chat_Va_Do_Tinh_Khiet",
        "TRO": "Tap_Chat_Va_Do_Tinh_Khiet", "KIM LOẠI": "Tap_Chat_Va_Do_Tinh_Khiet",
        "DUNG MÔI": "Tap_Chat_Va_Do_Tinh_Khiet", "ENDOTOXIN": "Tap_Chat_Va_Do_Tinh_Khiet",
        "TIỆT KHUẨN": "Tap_Chat_Va_Do_Tinh_Khiet", "ĐỘ TRONG": "Tap_Chat_Va_Do_Tinh_Khiet",
        "TỶ TRỌNG": "Tap_Chat_Va_Do_Tinh_Khiet", "GÓC QUAY": "Dinh_Tinh", 
        "ĐỘ NHỚT": "Tinh_Chat", "ĐỘ MỊN": "Tinh_Chat"
    }
    
    # Danh sách các Header CHÍNH (những mục này không cần ghi nhãn lại)
    MAIN_HEADERS = ["TÍNH CHẤT", "ĐỊNH TÍNH", "ĐỊNH LƯỢNG", "BẢO QUẢN", "LOẠI THUỐC", "HÀM LƯỢNG", "TẠP CHẤT", "ĐỘ HÒA TAN"]

    sorted_headers = sorted(headers_routing.keys(), key=len, reverse=True)

    for drug_chunk in raw_drugs:
        drug_chunk = drug_chunk.strip()
        if not drug_chunk: continue

        all_lines = [line.strip() for line in drug_chunk.split('\n') if line.strip()]
        lines = [line for line in all_lines if not is_image_line(line)]
        
        if not lines: continue

        # Tên thuốc
        raw_name_line = lines[0]
        clean_name = re.sub(r'^\d+(\.\d+)+\.?\s*', '', raw_name_line).strip()
        clean_name = clean_name.replace('</break>', '').strip()
        
        if len(clean_name) < 2: continue

        current_drug = {col: "" for col in core_columns}
        current_drug["Ten_Hoat_Chat"] = clean_text(clean_name).upper()
        
        current_section = "Mo_Ta_Chung"

        for i in range(1, len(lines)):
            line = lines[i]
            upper_line = line.upper()
            
            # 1. Tìm Tên Latin (Dòng 2)
            if i <= 2 and len(line) < 100 and not line.isupper() and not any(upper_line.startswith(k) for k in sorted_headers):
                 if not current_drug["Ten_Latin"]:
                     current_drug["Ten_Latin"] = clean_text(line)
                     continue

            # 2. Tìm Header (SỬA LỖI LOGIC TẠI ĐÂY)
            found_header = False
            for key in sorted_headers:
                target_col = headers_routing[key]
                
                if upper_line.startswith(key):
                    # Kiểm tra tính hợp lệ của Header (tránh cắt nhầm chữ)
                    remainder = line[len(key):]
                    is_valid = False
                    if not remainder: is_valid = True
                    elif not remainder[0].isalpha(): is_valid = True # Ký tự tiếp theo không phải chữ cái -> OK
                    
                    if is_valid:
                        # Lấy nội dung
                        content = remainder.strip()
                        if content.startswith(tuple([":", ".", "-"])):
                            content = content[1:].strip()
                        content = clean_text(content)

                        # --- FIX QUAN TRỌNG: CẬP NHẬT CURRENT_SECTION ---
                        # Bất kể là Header chính hay phụ, ta đều phải chuyển Section về cột đích
                        current_section = target_col 
                        
                        if key in MAIN_HEADERS:
                            # Nếu là Header chính: Chỉ cần thêm nội dung
                            if content: current_drug[current_section] += content + " "
                        else:
                            # Nếu là Header phụ (VD: PH, NƯỚC): Thêm Nhãn + Nội dung
                            # Kiểm tra xem có bị lặp nhãn không (tránh [PH]: [PH]:)
                            if not current_drug[current_section].strip().endswith(f"[{key}]:"):
                                labeled_content = f"[{key}]: {content} " if content else f"[{key}]: "
                                current_drug[current_section] += labeled_content
                            else:
                                # Nếu nhãn đã có rồi, chỉ thêm nội dung
                                current_drug[current_section] += content + " "

                        found_header = True
                        break 
            
            # Nếu dòng này không chứa Header mới -> Ghi vào Section hiện tại
            if not found_header:
                current_drug[current_section] += clean_text(line) + " "

        # Trích xuất công thức
        if not current_drug['Cong_Thuc_Hoa_Hoc']:
            search_text = current_drug['Mo_Ta_Chung'] + " " + current_drug['Ham_Luong_Yeu_Cau']
            current_drug['Cong_Thuc_Hoa_Hoc'] = extract_chemical_formula(search_text)

        data.append(current_drug)
    
    return pd.DataFrame(data)

# --- 3. HÀM GỘP FILE ---

def merge_all_files(list_docx_files, output_csv):
    all_data = []
    
    for file_path in list_docx_files:
        data_from_file = parse_single_docx(file_path) # Đổi tên gọi hàm cho khớp
        all_data.extend(data_from_file.to_dict('records'))
        
    df = pd.DataFrame(all_data)
    
    # Làm sạch & Điền khuyết
    cols_order = [
        "Ten_Hoat_Chat", "Ten_Latin", "Cong_Thuc_Hoa_Hoc", "Mo_Ta_Chung",
        "Tinh_Chat", "Dinh_Tinh", "Dinh_Luong", "Bao_Quan", 
        "Loai_Thuoc", "Ham_Luong_Yeu_Cau", "Tap_Chat_Va_Do_Tinh_Khiet", 
        "Do_Hoa_Tan"
    ]
    
    # Đảm bảo đủ cột
    for col in cols_order:
        if col not in df.columns: df[col] = ""
    df = df[cols_order]
    
    # Clean text
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        
    df.replace("", "không có thông tin", inplace=True)
    df.replace("nan", "không có thông tin", inplace=True)
    df.fillna("không có thông tin", inplace=True)
    
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print("="*40)
    print(f"Đã xử lý xong {len(df)} thuốc.")
    print(f"File lưu tại: {os.path.abspath(output_csv)}")

# Wrapper để khớp với gọi hàm cũ
def parse_single_docx(path):
    return parse_docx_to_df(path)

if __name__ == "__main__":
    current_script_path = os.path.abspath(__file__)
    kgraph_dir = os.path.dirname(current_script_path)
    preprocessing_dir = os.path.dirname(kgraph_dir)
    project_root = os.path.dirname(preprocessing_dir)

    data_folder = os.path.join(project_root, "data")

    files = [
        os.path.join(data_folder, "data-1-200.docx"),      
        os.path.join(data_folder, "data-201-615.docx"),
        os.path.join(data_folder, "data-616-815.docx")
    ]
    
    output = os.path.join(data_folder, "data_midterm.csv")
    
    print(f"--- Kiểm tra đường dẫn ---")
    print(f"Thư mục gốc dự án: {project_root}")
    print(f"Thư mục chứa data: {data_folder}")
    print(f"--------------------------")

    if not os.path.exists(data_folder):
        print(f"LỖI: Vẫn không tìm thấy thư mục: {data_folder}")
        print("Hãy đảm bảo thư mục 'data' nằm cùng cấp với thư mục 'preprocessing'.")
    else:
        merge_all_files(files, output)
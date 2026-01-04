from py2neo import Graph, Node, Relationship
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# ==========================================
# 1. CẤU HÌNH & KẾT NỐI
# ==========================================
# Kết nối Neo4j
try:
    # Lưu ý: Dùng bolt://127.0.0.1 cho kết nối ổn định trên máy cá nhân
    graph = Graph("neo4j://127.0.0.1:7687", auth=("neo4j", "12345678"))
    print("✅ Đã kết nối Neo4j thành công!")
except Exception as e:
    print(f"❌ Lỗi kết nối Neo4j: {e}")
    exit()

def clear_graph():
    """Xóa toàn bộ dữ liệu cũ trong Database"""
    print("⏳ Đang xóa dữ liệu cũ...")
    query = "MATCH (n) DETACH DELETE n"
    graph.run(query)
    print("✅ Đã xóa sạch Graph!")

# ==========================================
# 2. XỬ LÝ DỮ LIỆU
# ==========================================
def clean_text(text):
    """Làm sạch dữ liệu: Xử lý nan/null/không có thông tin"""
    if pd.isna(text) or text is None:
        return None
    text = str(text).strip()
    if text.lower() in ['không có thông tin', 'nan', '']:
        return None
    return text

def process_row(row):
    """Hàm xử lý từng dòng trong CSV"""
    try:
        # 1. Lấy thông tin cơ bản của HOẠT CHẤT
        ten_hoat_chat = clean_text(row.get('Ten_Hoat_Chat'))
        
        # Nếu không có tên hoạt chất thì bỏ qua dòng này
        if not ten_hoat_chat:
            return

        ten_latin = clean_text(row.get('Ten_Latin'))
        cong_thuc = clean_text(row.get('Cong_Thuc_Hoa_Hoc'))
        mo_ta = clean_text(row.get('Mo_Ta_Chung'))
        tinh_chat = clean_text(row.get('Tinh_Chat'))
        bao_quan = clean_text(row.get('Bao_Quan'))
        
        # 2. Tạo Node HOẠT_CHẤT
        hoat_chat_node = Node("HOẠT_CHẤT", 
                              tên_hoạt_chất=ten_hoat_chat,
                              tên_latin=ten_latin,
                              công_thức_hóa_học=cong_thuc,
                              mô_tả=mo_ta,
                              tính_chất=tinh_chat,
                              bảo_quản=bao_quan)
        graph.merge(hoat_chat_node, "HOẠT_CHẤT", "tên_hoạt_chất")

        # 3. Xử lý LOẠI THUỐC (Tạo node riêng để dễ truy vấn nhóm thuốc)
        loai_thuoc = clean_text(row.get('Loai_Thuoc'))
        if loai_thuoc:
            # Tách nếu có nhiều loại (ví dụ ngăn cách bởi dấu phẩy, tuỳ dữ liệu)
            # Ở đây giả sử mỗi dòng là 1 chuỗi mô tả loại thuốc
            category_node = Node("LOẠI_THUỐC", tên_loại=loai_thuoc)
            graph.merge(category_node, "LOẠI_THUỐC", "tên_loại")
            
            # Tạo quan hệ: Hoạt chất -> Thuộc nhóm -> Loại thuốc
            rel_cat = Relationship(hoat_chat_node, "THUỘC_NHÓM", category_node)
            graph.merge(rel_cat)

        # 4. Xử lý THÔNG TIN KIỂM NGHIỆM/TIÊU CHUẨN
        # Gom các trường kỹ thuật dài vào 1 node TIÊU_CHUẨN để Node chính đỡ nặng
        dinh_tinh = clean_text(row.get('Dinh_Tinh'))
        dinh_luong = clean_text(row.get('Dinh_Luong'))
        ham_luong = clean_text(row.get('Ham_Luong_Yeu_Cau'))
        tap_chat = clean_text(row.get('Tap_Chat_Va_Do_Tinh_Khiet'))
        do_hoa_tan = clean_text(row.get('Do_Hoa_Tan'))

        # Chỉ tạo node tiêu chuẩn nếu có ít nhất 1 thông tin
        if any([dinh_tinh, dinh_luong, ham_luong, tap_chat, do_hoa_tan]):
            tieu_chuan_node = Node("TIÊU_CHUẨN",
                                   thuộc_về_hoạt_chất=ten_hoat_chat, # Key để merge
                                   hàm_lượng_yêu_cầu=ham_luong,
                                   định_tính=dinh_tinh,
                                   định_lượng=dinh_luong,
                                   tạp_chất_và_độ_tinh_khiết=tap_chat,
                                   độ_hòa_tan=do_hoa_tan)
            graph.merge(tieu_chuan_node, "TIÊU_CHUẨN", "thuộc_về_hoạt_chất")
            
            # Tạo quan hệ: Hoạt chất -> Có tiêu chuẩn -> Tiêu chuẩn
            rel_std = Relationship(hoat_chat_node, "CÓ_TIÊU_CHUẨN", tieu_chuan_node)
            graph.merge(rel_std)

    except Exception as e:
        print(f"⚠️ Lỗi xử lý dòng {row.get('Ten_Hoat_Chat', 'Unknown')}: {e}")

# ==========================================
# 3. CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # 1. Xóa dữ liệu cũ
    clear_graph()

    # 2. Đọc file CSV
    # LƯU Ý: Thay đổi đường dẫn file CSV nếu cần
    csv_path = r'..\..\data\data_midterm.csv'  
    
    try:
        print(f"⏳ Đang đọc file CSV từ: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Kiểm tra xem các cột có đúng tên không
        expected_columns = ['Ten_Hoat_Chat', 'Ten_Latin', 'Cong_Thuc_Hoa_Hoc', 
                            'Mo_Ta_Chung', 'Tinh_Chat', 'Dinh_Tinh', 'Dinh_Luong', 
                            'Bao_Quan', 'Loai_Thuoc', 'Ham_Luong_Yeu_Cau', 
                            'Tap_Chat_Va_Do_Tinh_Khiet', 'Do_Hoa_Tan']
        
        # In ra các cột thực tế để debug nếu lỗi
        # print("Columns in CSV:", df.columns.tolist())

        print(f"📂 Tìm thấy {len(df)} dòng dữ liệu.")
        
        # 3. Chạy import song song
        # Giảm số worker xuống 1 nếu máy yếu hoặc gặp lỗi Lock Database
        num_workers = 4 
        print("🚀 Bắt đầu nạp dữ liệu vào Neo4j...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_row, row) for index, row in df.iterrows()]
            
            # Thanh tiến trình đơn giản
            count = 0
            total = len(df)
            for future in as_completed(futures):
                count += 1
                if count % 10 == 0:
                    print(f"   ...Đã xử lý {count}/{total} dòng")
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Lỗi thread: {e}")

        print("✅ HOÀN THÀNH NẠP DỮ LIỆU!")

    except FileNotFoundError:
        print(f"❌ Không tìm thấy file CSV tại: {csv_path}")
        print("👉 Hãy chắc chắn bạn đã lưu file dữ liệu mới và sửa đường dẫn trong code.")
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {e}")
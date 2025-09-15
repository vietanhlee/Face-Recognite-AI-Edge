import numpy as np
import cv2
import conf
import json
import os
import faiss

from PIL import ImageFont, ImageDraw, Image

def normalize_input(img: np.ndarray) -> np.ndarray:
    """Chuẩn hoá giá trị điểm ảnh về khoảng xấp xỉ [-1, 1].

    Thực hiện phép biến đổi (img - 127.5) / 128.0 phù hợp với nhiều
    mô hình nhận diện khuôn mặt họ Facenet.

    Args:
        img: Ảnh đầu vào dạng mảng có shape (H, W, C) và dtype là số thực hoặc số nguyên.

    Returns:
        Mảng đã được chuẩn hoá cùng shape với đầu vào.
    """
    # Các lựa chọn khác: chuẩn hoá theo mean/std của ảnh (đã giữ lại tham khảo dưới đây)
    # mean, std = img.mean(), img.std()
    # img = (img - mean) / std
    img = (img - 127.5) / 128.0
    return img

def read_image(path: str) -> np.ndarray:
    """Đọc ảnh từ đường dẫn và chuyển sang RGB.

    Args:
        path: Đường dẫn tuyệt đối hoặc tương đối đến file ảnh.

    Returns:
        Ảnh ở không gian màu RGB (ndarray, shape (H, W, 3)).

    Raises:
        ValueError: Nếu không đọc được ảnh tại ``path``.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # chuẩn RGB

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Tính độ tương tự cosine giữa hai vector đã được chuẩn hoá L2.

    Hàm này giả định ``v1`` và ``v2`` đã được L2-normalize. Khi đó
    ``np.dot(v1, v2)`` chính là cosine similarity.

    Args:
        v1: Vector đặc trưng thứ nhất, đã chuẩn hoá L2.
        v2: Vector đặc trưng thứ hai, đã chuẩn hoá L2.

    Returns:
        Giá trị độ tương tự cosine trong khoảng [-1, 1].
    """
    return np.dot(v1, v2)

def draw_box_text(img: np.ndarray, box: list, text: str, font_path="arial.ttf") -> np.ndarray:
    """
    Vẽ bounding box và text (hỗ trợ Unicode/tiếng Việt) lên ảnh.

    Args:
        img (np.ndarray): Ảnh gốc (OpenCV - BGR) sẽ bị sửa đổi tại chỗ.
        box (list): Toạ độ bounding box theo định dạng [x1, y1, x2, y2].
        text (str): Chuỗi muốn hiển thị (có thể là tiếng Việt/Unicode).
        font_path (str): Đường dẫn đến font .ttf hỗ trợ Unicode.

    Returns:
        Ảnh sau khi vẽ khung và text (BGR).
    """
    x1, y1, x2, y2 = box

    # Vẽ bounding box bằng OpenCV
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Chuyển sang PIL để vẽ text Unicode
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Load font hỗ trợ Unicode
    font = ImageFont.truetype(font_path, 18)

    # Vẽ text
    draw.text((x1, y1 - 25), text, font=font, fill=(0, 255, 0))

    # Chuyển ngược về OpenCV
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img


def add_id_name(id: int, name: str):
    """Thêm hoặc cập nhật ánh xạ id → tên vào file JSON cấu hình.

    Args:
        id: Định danh người dùng/đối tượng.
        name: Tên hiển thị tương ứng.
    """
    path = conf.path_json_id_name
    data = json.load(open(path, 'r', encoding='utf-8'))
    data[id] = name
    json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii= False, indent=4)

def delete_id_name(id: int, path: str = conf.path_json_id_name):
    """Xoá ánh xạ theo ``id`` khỏi file JSON nếu tồn tại.

    Args:
        id: Định danh cần xoá.
        path: Đường dẫn file JSON lưu ánh xạ id → tên.
    """
    id = str(id)
    data = json.load(open(path, 'r', encoding='utf-8'))
    if id in data.keys():
        del data[id]
    json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii= False, indent=4)

def check_is_id_exist(id: int, path: str = conf.path_json_id_name) -> bool:
    """Kiểm tra id có tồn tại trong file ánh xạ hay không.

    Args:
        id: Định danh cần kiểm tra.
        path: Đường dẫn file JSON lưu ánh xạ id → tên.

    Returns:
        True nếu tồn tại, ngược lại False.
    """
    data = json.load(open(path, 'r', encoding='utf-8'))
    return str(id) in data.keys() # Ép sang str vì json nó đọc ra str

def init_id_name(path: str = conf.path_json_id_name) -> dict:
    """Khởi tạo và đọc file ánh xạ id → tên.

    Nếu file chưa tồn tại, hàm sẽ tạo file rỗng rồi trả về dict trống.

    Args:
        path: Đường dẫn file JSON lưu ánh xạ id → tên.

    Returns:
        Dict ánh xạ id (chuỗi) → tên (chuỗi).
    """
    map_id_name = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            map_id_name = json.load(f)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(map_id_name, f)
    return map_id_name

def init_vt_db(path: str = conf.path_vector_db) -> faiss.Index:
    """Khởi tạo hoặc tải cơ sở dữ liệu vector FAISS.

    - Nếu file index tồn tại tại ``path``, tiến hành load.
    - Nếu không tồn tại, tạo mới ``IndexIDMap2(IndexFlatIP(dim))`` và lưu.

    Args:
        path: Đường dẫn đến file index FAISS.

    Returns:
        Đối tượng ``faiss.Index`` đã sẵn sàng cho tìm kiếm/thêm vector.
    """
    if os.path.exists(path):
        print("Đang load vector db")
        index = faiss.read_index(path)
        print("Đã load xong vector bd")
    else:
        print("Không tìm thấy vector db, chuẩn bị tạo mới")
        # Sử dụng IndexFlatIP để tìm kiếm tương tự cos
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(conf.dim)) 
        faiss.write_index(index, path)
        print("Đã tạo xong vector bb")
    return index


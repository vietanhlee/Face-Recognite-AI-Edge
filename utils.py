import numpy as np
import cv2
import conf
import json
import os
import faiss

from PIL import ImageFont, ImageDraw, Image

def normalize_input(img: np.ndarray) -> np.ndarray:
    # mean, std = img.mean(), img.std()
    # img = (img - mean) / std
    img = (img - 127.5) / 128.0
    return img

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # chuẩn RGB

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2)

def draw_box_text(img: np.ndarray, box: list, text: str, font_path="arial.ttf") -> np.ndarray:
    """
    Vẽ bounding box và text (có hỗ trợ tiếng Việt) lên ảnh.

    Args:
        img (np.ndarray): ảnh gốc (OpenCV format - BGR).
        box (list): [x1, y1, x2, y2] toạ độ bounding box.
        text (str): chuỗi tiếng Việt muốn hiển thị.
        font_path (str): đường dẫn đến file font .ttf hỗ trợ tiếng Việt.
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
    path = conf.path_json_id_name
    data = json.load(open(path, 'r', encoding='utf-8'))
    data[id] = name
    json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii= False, indent=4)

def delete_id_name(id: int, path: str = conf.path_json_id_name):
    id = str(id)
    data = json.load(open(path, 'r', encoding='utf-8'))
    if id in data.keys():
        del data[id]
    json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii= False, indent=4)

def check_is_id_exist(id: int, path: str = conf.path_json_id_name) -> bool:
    data = json.load(open(path, 'r', encoding='utf-8'))
    return str(id) in data.keys() # Ép sang str vì json nó đọc ra str

def init_id_name(path: str = conf.path_json_id_name) -> dict:
    map_id_name = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            map_id_name = json.load(f)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(map_id_name, f)
    return map_id_name

def init_vt_db(path: str = conf.path_vector_db) -> faiss.Index:
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


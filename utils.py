from matplotlib.pyplot import box
import numpy as np
import cv2

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


def verify_face(embed1: np.ndarray, embed2: np.ndarray, threshold: float = 0.7) -> dict:
    sim = round(float(cosine_similarity(embed1, embed2)), 3)
    return {
        "similarity": sim,
        "verified": sim > threshold
    }

def draw_box_text(img: np.ndarray, box: list, text: str) -> np.ndarray:
    (x, y, w, h), text = (box, text)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img
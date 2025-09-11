from keras.models import load_model
import numpy as np
from FaceDetectYolo import FaceDetectYolo
from utils import *
from vector_db import VectorBD
import conf

class Regconizer():
    def __init__(self, model_path: str = conf.path_model_face_recognition):
        self.model = load_model(model_path, compile=False, safe_mode=False)
        self.img_face = None
        self.img_with_bb = None
        self.vt_db = VectorBD()
        self.detector_face = FaceDetectYolo()
        # self.detector_face = FaceDetect()
        
    def get_face_embedding(self, img: np.ndarray) -> np.ndarray:
        """
        Trả về embedding đã chuẩn hóa L2, luôn có shape (N, D).
        - N = số mặt detect được (>=1)
        - D = chiều embedding
        """
        self.detector_face.set_img_input(img)
        # Lấy batch ảnh khuôn mặt (N, H, W, C)
        faces = self.detector_face.cropped_faces
    
        if faces is None or len(faces) == 0:
            return np.array([])

        # Model predict -> (N, D) nếu nhiều ảnh, hoặc (D,) nếu 1 ảnh
        embeds = self.model.predict(faces, verbose=False)


        # Chuẩn hóa L2 theo từng vector
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / norms

        return embeds  

    def regcognize_face(self, img: np.ndarray) -> dict:
        embed = self.get_face_embedding(img)
        if embed is None or embed.size == 0:
            return {
                "Distances": [],
                "Names": []
            }
        dis, name = self.vt_db.search_emb(embed)
        return {
            "Distances": dis,
            "Names": name
        }
        
    def register_face(self, img: np.ndarray, name: str):
        embed = self.get_face_embedding(img)[0]
        self.vt_db.update_emb(np.expand_dims(embed, axis=0), name)
        print(f"Đã đăng ký gương mặt với tên: {name}")
    

from keras.models import load_model
import numpy as np
from FaceDetectYolo import FaceDetectYolo
from utils import *
from vector_db import VectorBD
import conf
from tensorflow.lite.python.interpreter import Interpreter

class Regconizer():
    def __init__(self, model_path: str = conf.path_model_face_recognition):
        # self.model = load_model(model_path, compile=False, safe_mode=False)
        self.img_face = None
        self.img_with_bbs = None
        self.vt_db = VectorBD()
        self.detector_face = FaceDetectYolo()
        # self.detector_face = FaceDetect()
        self.bbs = []
        # ===== Load model TFLite =====
        self.interpreter = Interpreter(model_path= model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_face_embedding(self, img: np.ndarray) -> np.ndarray:
        """
        Trả về embedding đã chuẩn hóa L2, luôn có shape (N, D).
        - N = số mặt detect được (>=1)
        - D = chiều embedding
        """
        self.detector_face.set_img_input(img)
        
        # Lấy faces ảnh khuôn mặt (N, H, W, C)
        faces = self.detector_face.cropped_faces
        self.bbs = self.detector_face.bbs  # Lưu bounding box
        
        if faces is None or len(faces) == 0:
            return np.array([])

        # Model predict -> (N, D) nếu nhiều ảnh, hoặc (D,) nếu 1 ảnh
        # embeds = self.model.predict(faces, verbose=False)

        if faces.dtype != self.input_details[0]['dtype']:
            faces = faces.astype(self.input_details[0]['dtype'])

        # Set tensor & chạy infer
        self.interpreter.resize_tensor_input(self.input_details[0]['index'], faces.shape)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], faces)
        self.interpreter.invoke()

        # Lấy output
        embeds = self.interpreter.get_tensor(self.output_details[0]['index'])  # (faces_size, 128)

        # Chuẩn hóa L2 theo từng vector
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / norms

        return embeds  

    def regcognize_face(self, img: np.ndarray) -> dict:
        embed = self.get_face_embedding(img)

        if embed is None or len(embed) == 0:
            return {
                "Distances": [],
                "Names": []
            }

        distances, names = self.vt_db.search_emb(embed)

        for dist, name, bb in zip(distances, names, self.bbs):
            if dist[0] > conf.threshold_distance:
                text = f"{name[0]} ({dist[0]:.2f})"
            else:
                text = "Unknown"
            self.img_with_bbs = draw_box_text(img, bb, text)
        
        
        return {
            "Distances": distances,
            "Names": names
        }
        
    def register_face(self, img: np.ndarray, name: str):
        embed = self.get_face_embedding(img)
        if len(embed) > 1:
            print("Có nhiều hơn 1 khuôn mặt trong ảnh. Vui lòng sử dụng ảnh chỉ có 1 khuôn mặt.")
            return
        self.vt_db.update_emb(embed, name)
        print(f"Đã đăng ký gương mặt với tên: {name}")
    

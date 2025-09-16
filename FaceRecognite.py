# from keras.models import load_model
import numpy as np
from FaceDetectYolo import FaceDetectYolo
from utils import *
from VectorDB import VectorBD
import conf
from tensorflow.lite.python.interpreter import Interpreter

class Regconizer():
    """Bao bọc pipeline nhận diện: detect → embed → search.

    Args:
        model_path: Đường dẫn mô hình TFLite nhận diện khuôn mặt.
    """
    def __init__(self, model_path: str = conf.path_model_face_recognition):
        # self.model = load_model(model_path, compile=False, safe_mode=False)
        self.img_face = None
        self.img_with_bbs = None
        self.vt_db = VectorBD()
        self.detector_face = FaceDetectYolo()
        # self.detector_face = FaceDetect()
        self.bbs = []
        
        self.interpreter = Interpreter(model_path= model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_face_embedding(self, img: np.ndarray) -> np.ndarray:
        """Suy luận embedding cho các khuôn mặt trong ảnh.

        Args:
            img (np.ndarray): Ảnh đầu vào (BGR - OpenCV).

        Returns:
            np.ndarray: Batch embedding có shape (N, D). Nếu không có mặt, trả về mảng rỗng.
        """
        # Set ảnh đầu vào 
        self.detector_face.set_img_input(img)
        
        # Lấy faces ảnh khuôn mặt (N, H, W, C)
        faces = self.detector_face.cropped_faces
        self.bbs = self.detector_face.bbs_face # Lưu bounding box
        
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

        # Lấy output (đầu vào dạng batch)
        embeds = self.interpreter.get_tensor(self.output_details[0]['index'])  # (faces_size, 128)

        # Chuẩn hóa L2 theo từng vector
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / norms

        return embeds  

    def regcognize_face(self, img: np.ndarray) -> dict:
        """Nhận diện các khuôn mặt, trả về kết quả tìm kiếm và ảnh có vẽ nhãn.

        Args:
            img (np.ndarray): Ảnh đầu vào (BGR - OpenCV).

        Returns:
            dict: Bao gồm khoảng cách, tên, ids 
        """
        embeddings = self.get_face_embedding(img)
        self.img_with_bbs = img
        
        if embeddings is None or len(embeddings) == 0:
            return {
                "Distances": [],
                "Names": [],
                "IDs": [],
            }
        
        # Tìm kiếm cho embeddings
        distances, names, ids = self.vt_db.search_emb(embeddings)
        
        for dist, name, bb, id in zip(distances, names, self.bbs, ids):
            # Chọn nhãn cho bbs
            if dist[0] > conf.threshold_distance:
                text = f"{id[0]} - {name[0]} ({dist[0]:.2f})"
            else:
                text = "Unknown"
            
            # Vẽ bbs và nhãn lên màn
            self.img_with_bbs = draw_box_text(self.img_with_bbs, bb, text)
        
        return {
            "Distances": distances,
            "Names": names, 
            "IDs": ids
        }

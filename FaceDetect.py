from mtcnn import MTCNN
import numpy as np
import cv2
from utils import *
from conf import *

class FaceDetect():
    def __init__(self):
        self.detector = MTCNN()
        self.imgs_face_best_for_rec = None
        self.img_with_bbs = None
        self.count_face = 0
        self.bbs = []
        self.cropped_faces = np.array([])
        
    def set_img_input(self, img: np.ndarray, target_size=(160, 160)) -> np.ndarray:
        self.img_with_bbs = img.copy()
        faces = self.detector.detect_faces(img)
        self.count_face = len(faces)

        if len(faces) == 0:
            print("Không phát hiện thấy gương mặt nào")
            
        self.cropped_faces = np.array([])
        self.img_faces = []  # lưu ảnh crop gốc (không resize)
        self.bbs = []
        for idx, face in enumerate(faces):
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            self.bbs.append((x, y, w, h))
            cropped = img[y : y + h, x : x + w]
            self.img_faces.append(cropped.copy())  # lưu ảnh gốc
            if idx == 0:
                self.imgs_face_best_for_rec = cropped.copy()                
            # Vẽ bounding box
            cv2.rectangle(self.img_with_bbs, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Resize về target size
            cropped = cv2.resize(cropped, target_size)

            # Scale về [0,1] rồi chuẩn hóa cho model
            cropped = cropped.astype("float32")
            cropped = normalize_input(cropped)

            self.cropped_faces = np.append(self.cropped_faces, cropped)


if __name__ == '__main__':
    import datetime
    timenow = datetime.datetime.now()
    
    cam = cv2.VideoCapture(0)
    detector = FaceDetect()
    while True:
        
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        detector.set_img_input(frame)

        
        time_pre = datetime.datetime.now()
        fps = 1 / (time_pre - timenow).total_seconds()
        timenow = time_pre
        cv2.putText(detector.img_with_bbs, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Detection", detector.img_with_bbs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()
from ultralytics import YOLO
import numpy as np
import cv2
from utils import *
import conf

class FaceDetectYolo():
    def __init__(self, model_path: str = conf.path_model_face_detection):
        self.detector = YOLO(model_path)
        self.imgs_face_best_for_rec = None
        self.img_with_bbs = None
        self.count_face = 0
        self.bbs = []
        self.cropped_faces = np.array([])

    def set_img_input(self, img: np.ndarray, target_size=(160, 160)) -> np.ndarray:
        self.img_with_bbs = img.copy()
        results = self.detector.predict(img, verbose=False)
        faces = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
        self.count_face = len(faces)

        if len(faces) == 0:
            print("Không phát hiện thấy gương mặt nào")

        self.cropped_faces = np.array([])
        
        self.bbs = []
        for idx, box in enumerate(faces):
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            w, h = x2 - x1, y2 - y1
            self.bbs.append((x1, y1, w, h))
            cropped = img[y1 : y2, x1 : x2]
            
            cv2.rectangle(self.img_with_bbs, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
            cropped_resized = cv2.resize(cropped, target_size)
            cropped_resized = normalize_input(cropped_resized)
            if idx == 0:
                self.imgs_face_best_for_rec = cropped.copy()
           
            if self.cropped_faces.size == 0:
                self.cropped_faces = np.expand_dims(cropped_resized, axis=0)
            else:
                self.cropped_faces = np.append(self.cropped_faces, [cropped_resized], axis=0)


if __name__ == '__main__':
    import datetime
    timenow = datetime.datetime.now()

    cam = cv2.VideoCapture(0)
    detector = FaceDetectYolo()
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        detector.set_img_input(frame)
        print(detector.cropped_faces.shape)
        time_pre = datetime.datetime.now()
        fps = 1 / (time_pre - timenow).total_seconds()
        timenow = time_pre
        cv2.putText(detector.img_with_bbs, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Detection", detector.img_with_bbs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
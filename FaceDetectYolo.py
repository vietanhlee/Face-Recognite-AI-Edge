from ultralytics import YOLO
import numpy as np
import cv2
from utils import *
import conf

class FaceDetectYolo():
    def __init__(self, model_path: str = conf.path_model_face_detection):
        """Khởi tạo detector YOLO và các trường kết quả."""
        self.detector = YOLO(model_path, task="detect")  # Mô hình YOLO để phát hiện khuôn mặt
        self.img_with_bbs = None          # Ảnh đầu vào kèm bounding boxes để hiển thị
        self.cropped_faces = np.array([]) # Batch ảnh khuôn mặt đã resize + normalize cho mô hình nhận diện
        self.bbs_face = []                # Danh sách bbox khuôn mặt theo định dạng [x1, y1, x2, y2]
    def set_img_input(self, img: np.ndarray, target_size=(160, 160)) -> None:
        """Chạy phát hiện khuôn mặt trên ảnh và chuẩn bị batch đầu vào.

        Args:
            img (np.ndarray): Ảnh đầu vào (BGR - OpenCV).
            target_size (tuple): Kích thước HxW cho ảnh khuôn mặt đầu vào model nhận diện.
        """
        self.img_with_bbs = img.copy()
        results = self.detector.predict(img, verbose=False)
    
        # Cần đảm bảo array nằm trên cpu để tính toán với các hàm
        self.bbs_face = results[0].boxes.xyxy.cpu().numpy().astype(dtype= np.int64) 
        
        self.cropped_faces = np.array([])
        
        for box in self.bbs_face:
            x1, y1, x2, y2 = box
            # Cắt ảnh            
            cropped = img[y1 : y2, x1 : x2]
            
            # Vẽ box lên đầu ra
            cv2.rectangle(self.img_with_bbs, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Tiền xử lý ảnh
            cropped_resized = cv2.resize(cropped, target_size)
            cropped_resized = normalize_input(cropped_resized)
            
            # Bước thêm các ảnh sau xử lý vào array kiểm tra array có rỗng không mới dùng append được
            if len(self.cropped_faces) == 0:
                self.cropped_faces = np.expand_dims(cropped_resized, axis=0)
            else:
                self.cropped_faces = np.append(self.cropped_faces, [cropped_resized], axis=0)


#******************************************** Code test **********************************************
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
        
        time_pre = datetime.datetime.now()
        fps = 1 / (time_pre - timenow).total_seconds()
        timenow = time_pre
        cv2.putText(detector.img_with_bbs, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Detection", detector.img_with_bbs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
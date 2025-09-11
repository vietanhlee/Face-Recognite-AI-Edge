from FaceRecognite import Regconizer
import cv2
import datetime

if '__main__' == __name__:
    # thêm fps
    rcg = Regconizer()
    cam = cv2.VideoCapture(0)
    timenow = datetime.datetime.now()
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Đảo ảnh
        res = rcg.regcognize_face(frame)
        frame = rcg.detector_face.img_with_bbs
        # Chỉ log nếu có khuôn mặt và distance >= 0.7
        if len(res['Names']) > 0 and len(res['Distances']) > 0 and res['Distances'][0][0] >= 0.7:
            print(res['Names'], res['Distances'])
        time_pre = datetime.datetime.now()
        fps = 1 / (time_pre - timenow).total_seconds()
        timenow = time_pre
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
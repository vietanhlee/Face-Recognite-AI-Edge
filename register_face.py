"""
Đăng ký khuôn mặt bằng cách chụp 9 hướng cố định và lưu embedding.
"""

import cv2
import os
import numpy as np
import shutil
import argparse
from FaceRecognite import Regconizer
from VectorDB import VectorBD
from utils import check_is_id_exist

def main(name: str, id: int):
    os.makedirs('images', exist_ok=True)

    if check_is_id_exist(id):
        print(f"❌ ID {id} đã tồn tại trong database. Hãy chọn ID khác.")
        return

    # Tạo thư mục lưu ảnh
    dir_path = f'./images/{id}_{name}'
    os.makedirs(dir_path, exist_ok=True)

    # Thứ tự chụp các hướng (bao gồm chéo)
    directions = [
        'mid', 'left', 'right', 'up', 'down',
        'up_left', 'up_right', 'down_left', 'down_right'
    ]
    current_idx = 0

    # Khởi tạo nhận diện và DB
    rec = Regconizer()
    vt_db = VectorBD()

    # Mở camera
    cam = cv2.VideoCapture(0)

    def cleanup_and_exit(remove_folder=False):
        cam.release()
        cv2.destroyAllWindows()
        if remove_folder and os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"❌ Đã xoá folder {dir_path} vì phát hiện nhiều khuôn mặt.")
        print("Thoát chương trình")
        exit(0)

    print("👉 Nhìn theo thứ tự: mid → left → right → up → down | Nhấn: [P] chụp, [Q] thoát")

    embeddings_buffer = []

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Không đọc được khung hình từ camera")
            cleanup_and_exit(remove_folder=False)
        frame = cv2.flip(frame, 1)

        # Tính embedding và cập nhật ảnh có bbox để hiển thị
        embed = rec.get_face_embedding(frame)
        display_img = rec.detector_face.img_with_bbs if hasattr(rec, 'detector_face') else frame

        # Overlay UI: hướng hiện tại, tiến độ, hướng dẫn phím
        h, w = display_img.shape[:2]
        bar_w = int((current_idx / len(directions)) * w)
        cv2.rectangle(display_img, (0, h-10), (bar_w, h), (0, 255, 0), -1)
        cur_dir = directions[current_idx]
        cv2.putText(display_img, f"Direction: {cur_dir}  ({current_idx+1}/{len(directions)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(display_img, "[P] capture  [Q] quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # Hiển thị khung hình
        cv2.imshow("Camera", display_img)

        # Đọc phím một lần mỗi vòng
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cleanup_and_exit(remove_folder=False)
        if key == ord('p'):
            if len(embed) == 1:
                # Lưu ảnh theo hướng hiện tại
                dir_name = cur_dir
                img_path = f"{dir_path}/{dir_name}.jpg"
                cv2.imwrite(img_path, frame)

                # Lưu embedding
                embeddings_buffer.append(embed[0])
                print(f"Đã lưu {dir_name} → {img_path}")

                # Tiến hướng kế tiếp
                current_idx += 1
                if current_idx >= len(directions):
                    # Đủ 9 hướng → lưu DB
                    embeds = np.array(embeddings_buffer)
                    vt_db.add_emb(embeds, name, id)
                    print("✅ Hoàn tất đăng ký và lưu embeddings vào DB.")
                    cleanup_and_exit(remove_folder=False)
            elif len(embed) == 0:
                print("⚠️ Không phát hiện gương mặt nào. Hãy đưa mặt vào khung.")
            else:
                print("⚠️ Tồn tại nhiều hơn 1 gương mặt. Hãy đảm bảo chỉ có 1 người trong khung.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Tên người cần đăng ký")
    parser.add_argument("--id", type=int, required=True, help="ID người cần đăng ký")
    args = parser.parse_args()

    main(args.name, args.id)

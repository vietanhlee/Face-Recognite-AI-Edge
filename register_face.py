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

    # Các hướng cần chụp
    directions = ['mid', 'left', 'right', 'up', 'down']
    direction_iter = iter(directions)

    # Khởi tạo nhận diện và DB
    rec = Regconizer()
    vt_db = VectorBD()

    # Mở camera
    cam = cv2.VideoCapture(0)

    def exit_program(remove_folder=False):
        cam.release()
        cv2.destroyAllWindows()
        if remove_folder and os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"❌ Đã xoá folder {dir_path} vì phát hiện nhiều khuôn mặt.")
        print("Thoát chương trình")
        exit(0)

    print("👉 Hãy lần lượt nhìn: mid → left → right → up → down. Nhấn 'p' để chụp, 'q' để thoát.")

    embeds = []  # list chứa embedding các hướng

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # Lấy embedding từ frame
        embed = rec.get_face_embedding(frame)

        # Bấm 'p' để chụp ảnh
        if cv2.waitKey(1) & 0xFF == ord('p'):
            if len(embed) == 1:
                try:
                    dir_name = next(direction_iter)
                    img_path = f"{dir_path}/{dir_name}.jpg"
                    cv2.imwrite(img_path, frame)

                    embeds.append(embed[0])  # embed là list, lấy phần tử [0]
                    print(f"Đã lưu ảnh {dir_name} ({img_path}) và embedding tạm thời")

                except StopIteration:
                    # Sau khi chụp đủ 5 hướng → convert sang numpy và lưu vào DB
                    embeds = np.array(embeds)
                    vt_db.add_emb(embeds, name, id)
                    print("✅ Đã đủ ảnh, hoàn tất đăng ký khuôn mặt và lưu embeddings vào DB.")
                    exit_program()
            elif len(embed) == 0:
                print("⚠️ Không phát hiện gương mặt nào")
            else:
                print("⚠️ Tồn tại nhiều hơn 1 gương mặt")

        # Hiển thị khung hình với bounding box
        cv2.imshow("Camera", rec.detector_face.img_with_bbs)

        # Bấm 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_program()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Tên người cần đăng ký")
    parser.add_argument("--id", type=int, required=True, help="ID người cần đăng ký")
    args = parser.parse_args()

    main(args.name, args.id)

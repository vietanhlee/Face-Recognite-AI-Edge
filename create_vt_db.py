import os
import numpy as np
import argparse
from VectorDB import VectorBD
from FaceRecognite import Regconizer
from utils import read_image, check_is_id_exist

def add_emb_in_folder(root_folder: str, is_reinit: bool = True) -> None:
    vt = VectorBD()
    if is_reinit:
        # Khởi tạo database
        vt.re_init()
    
    reg = Regconizer()

    # duyệt từng folder con (mỗi người)
    for person_id, person_name in enumerate(os.listdir(root_folder)):
        person_folder = os.path.join(root_folder, person_name)
        
        # Nếu không reinit thì id là phần đằng trước của tên folder (folder = id_name)
        if not is_reinit:
            person_id = int(person_name.split('_')[0])
        person_name = person_name.split('_')[-1]

        if check_is_id_exist(person_id):
            print(f"⚠️ {person_id} của {person_name} đã tồn tại, bỏ qua.")
            continue
        
        person_embs = []
        for file in os.listdir(person_folder):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(person_folder, file)
                try:
                    img = read_image(path)
                    embed = reg.get_face_embedding(img)
                    
                    if len(embed) > 1:
                        print(f"⚠️ Ảnh {file} trong {person_name} có nhiều hơn 1 khuôn mặt, bỏ qua.")
                        continue
                    person_embs.append(embed[-1])
                except Exception as e:
                    print(f"❌ Lỗi với ảnh {file} trong {person_name}: {e}")

        if len(person_embs) > 0:
            person_embs = np.array(person_embs)
            vt.add_emb(person_embs, person_name, person_id)

    print("✅ Đã lưu xong tất cả embedding trong folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reinit", type=int, required=True, help="1 = Khởi tạo lại DB, 0 = Thêm mới vào DB")
    args = parser.parse_args()

    add_emb_in_folder(root_folder="./images", is_reinit=bool(args.reinit))

import os
import numpy as np
from vector_db import VectorBD
from FaceRecognite import Regconizer
from utils import read_image, check_is_id_exist

def add_emb_in_folder(root_folder: str, is_reinit: bool = True) -> None:
    """
    Khởi tạo lại hoặc thêm vector database từ nhiều folder ảnh
    Args:
        root_folder (str): Folder gốc, mỗi folder con là tên mỗi người
        is_reinit (bool): True có nghĩa là khởi tạo lại, False là thêm mới
    """
    vt = VectorBD()
    if is_reinit == True:
        # Khởi tạo database
        vt.re_init()
    
    reg = Regconizer()

    # duyệt từng folder con (mỗi người)
    for person_id, person_name in enumerate(os.listdir(root_folder)):
        person_folder = os.path.join(root_folder, person_name)
        
        # Nếu không reinit thì id là phần đằng trước của tên folder (folder = id_name)
        if is_reinit == False:
            person_id = int(person_name.split('_')[0])
        person_name = person_name.split('_')[-1]
        # Kiểm tra id đã tồn tại hay chưa, tồn tại thì tiếp đến cái khác
        if check_is_id_exist(person_id) == True:
            print(f"{person_id} của {person_name} này đã tồn tại, bỏ qua.")
            continue
        
        # List chứa embeddings gộp để thêm một thể cho tối ưu performance
        person_embs = []
        for file in os.listdir(person_folder):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(person_folder, file)
                try:
                    img = read_image(path)
                    embed = reg.get_face_embedding(img)
                    
                    # Nếu tồn tại nhiều hơn 1 mặt thì bỏ qua
                    if len(embed) > 1:
                        print(f"Cảnh báo: Ảnh {file} trong {person_name} có nhiều hơn 1 khuôn mặt, bỏ qua.")
                        continue
                    person_embs.append(embed[-1]) # Đầu ra có dạng [embed] vì nó predict theo batch
                except Exception as e:
                    print(f"Lỗi với ảnh {file} trong {person_name}: {e}")
        # Bước thêm embeddings một thể
        if len(person_embs) > 0:
            person_embs = np.array(person_embs) # Chuyển sang array do yêu cầu embeding của faiss
            vt.add_emb(person_embs, person_name, person_id)

    print("----------------------------Đã lưu xong tất cả embedding trong folder --------------------------------")

add_emb_in_folder(root_folder= "./images", is_reinit= True)
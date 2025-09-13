from vector_db import VectorBD
import os

def save_all_emb_in_folder(folder: str):
    """Khởi tạo vector database từ 1 folder ảnh
    Args:
        folder (str): Tên folder chứa ảnh (không cần thiết là ảnh gương mặt), tên file ảnh là tên của người đó
    """
    vt = VectorBD()
    vt.re_init()
    from FaceRecognite import Regconizer
    from utils import read_image
    reg = Regconizer()
    for file in os.listdir(folder):
        name = file.split('.')[0]
        if file.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(folder, file)
            try:
                img = read_image(path)
                embed = reg.get_face_embedding(img)
                if len(embed) > 1:
                    print(f"Cảnh báo: Ảnh {file} có nhiều hơn 1 khuôn mặt, bỏ qua ảnh này.")
                    continue
                vt.add_emb(embed, name)
            except Exception as e:
                print(f"Lỗi với ảnh {file}: {e}")
    print("Đã lưu xong tất cả embedding trong folder")
        
save_all_emb_in_folder('./images')  
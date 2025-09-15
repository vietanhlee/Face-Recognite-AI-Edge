from VectorDB import VectorBD
from utils import check_is_id_exist
import shutil
import os
import argparse

def delete_folder_id(id: int):
    id = str(id)
    folder = './images'
    for name in os.listdir(folder):
        if name.split('_')[0] == id:
            shutil.rmtree(os.path.join(folder, name))

def main(id: int):
    vt_db = VectorBD()
    if not check_is_id_exist(id):
        print(f"❌ ID {id} không tồn tại trong database.")
        return

    vt_db.remove_emb(id)
    delete_folder_id(id)
    print(f"✅ Đã xoá ID {id} khỏi database và thư mục ảnh.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="ID cần xoá")
    args = parser.parse_args()

    main(args.id)

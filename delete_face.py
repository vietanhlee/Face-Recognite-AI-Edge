from VectorDB import VectorBD
from utils import check_is_id_exist
import shutil
import os

def delete_folder_id(id: int):
    id = str(id)
    folder = './images'
    for name in os.listdir(folder):
        if name.split('_')[0] == id:
            shutil.rmtree(os.path.join(folder, name))
            
def delete_id():
    vt_db = VectorBD()
    id = ' '
    while True:
        id = int(input("Nhập ID cần xoá: "))
        if check_is_id_exist(id) == True:
            break

    vt_db.remove_emb(id)
    delete_folder_id(id)


delete_id()
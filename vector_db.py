import numpy as np
import faiss
import os
import conf

class VectorBD:
    def __init__(self, path_db : str = conf.path_vector_db, path_list_names : str = conf.path_list_names):
        self.dim = 128
        self.path_db = path_db
        self.path_list_names = path_list_names
        self.index = self.load_vt_db()
        self.list_names = self.load_list_names()
        
    def load_list_names(self):
        if os.path.exists(self.path_list_names):
            list_names = np.load(self.path_list_names, allow_pickle=True)
        else:
            list_names = np.array([])
            np.save(self.path_list_names, list_names)
        return list_names

    def load_vt_db(self):
        if os.path.exists(self.path_db):
            print("Đang load vector db")
            index = faiss.read_index(self.path_db)
            print("Đã load xong vector bd")
        else:
            print("Không tìm thấy vector db, chuẩn bị tạo mới")
            # index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
            index = faiss.IndexFlatIP(self.dim)  # Sử dụng IndexFlatIP để tìm kiếm tương tự cos
            faiss.write_index(index, self.path_db)
            print("Đã tạo xong vector bb")
        return index
    
    def search_emb(self, emb):
        dis, id = self.index.search(emb, 1)
        name = self.list_names[id]
        return dis, name
    
    def remove_emb(self, name: str):
        id = np.where(self.list_names == name)[0]
        if id is not None and id.shape[0] > 0:
            self.index.remove_ids(id)  
            self.list_names = np.delete(self.list_names, int(id[0]))
            self.save_local()
            print("Đã xoá thành công")
        else:
            print("Không tìm thấy tên trong db")
    
    def update_emb(self, emb, name: str):
        self.remove_emb(name)
        self.add_emb(emb, name)
        print("Đã cập nhật thành công")
        
    def save_local(self):
        np.save(self.path_list_names, self.list_names)
        faiss.write_index(self.index, self.path_db)
    
    def add_emb(self, emb, name: str):
        # self.index.add_with_ids(emb, id)
        id = np.where(self.list_names == name)[0]
        if id is not None and id.shape[0] == 0:
            self.index.add(emb)
            self.list_names = np.append(self.list_names, name)
            self.save_local()
            print(f"Đã thêm thành công {name} vào db")
        else:
            print(f"{name} đã tồn tại trong db, vui lòng sử dụng hàm cập nhật")

    def re_init(self):
        if os.path.exists(self.path_db):
            os.remove(self.path_db)
        if os.path.exists(self.path_list_names):
            os.remove(self.path_list_names)
        self.index = self.load_vt_db()
        self.list_names = self.load_list_names()
        print("Đã tạo lại db mới")
        
    
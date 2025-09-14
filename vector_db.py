import numpy as np
import faiss
import os
import conf
from utils import load_id_name, load_vt_db, delete_id_name, check_is_id_exist, add_id_name

class VectorBD:
    def __init__(self, path_db : str = conf.path_vector_db, path_json_id_name: str = conf.path_json_id_name):
        self.dim = conf.dim
        self.path_db = path_db
        self.path_json_id_name = path_json_id_name
        self.index = load_vt_db(self.path_db)
        self.map_id_name = load_id_name(self.path_json_id_name)

    def search_emb(self, embeddings: np.ndarray):
        """ Tìm ra vector có độ tương đồng cosin gần nhất với vector đưa vào

        Args:
            embeddings (np.ndarray): vector embedding đưa vào (định dạng batch). Ví dụ: [embed1, embed2]

        Returns:
            np.ndarray: đưa ra khoảng cách, tên, id tương ứng \n
            dis = [[dis1], [dis2], ...]
            names = [[name1], [name2], ...]
            ids = [[id1], [id2]]
        """
        dis, ids = self.index.search(embeddings, 1)
        names = [[self.map_id_name.get(str(id[0]), "Unknown")] for id in ids]
        return dis, names, ids
    
    def remove_emb(self, id: int):
        """ Xoá tất cả các embedding có id tương ứng là id

        Args:
            id (int): id của embedding cần xoá
        """
        self.index.remove_ids(np.array([id])) # Yêu cầu đầu vào là array
        self.save_local() # Xoá xong phải lưu lại database
        delete_id_name(id, self.path_json_id_name) # Xoá luôn ở file json map id --> name

    def update_emb(self, embeddings: np.ndarray, id: int):
        """ Cập nhật embedding theo id 

        Args:
            embeddings (np.ndarray): Các vector embedding (đingj dạng batch)
            id (int): id gắn với vector embeddings đó
        """
        self.remove_emb(id)
        self.add_emb(embeddings, id)
        print("Đã cập nhật thành công")
        
    def save_local(self):
        """ Lưu lại database
        """
        faiss.write_index(self.index, self.path_db)
    
    def add_emb(self, embeddings: np.ndarray, name: str, id: int):
        """ Hàm thêm embedding vào database 

        Args:
            embeddings (np.ndarray): các embedding (định dạng batch)
            name (str): tên của người đó
            id (int): id tương ứng 
        """
        # Kiểm tra id đã tồn tại hay chưa
        if not check_is_id_exist(id, self.path_json_id_name):
            # Thêm vào index embeddings với id tương ứng
            self.index.add_with_ids(embeddings, np.array([id] * len(embeddings)))
            add_id_name(id, name) # Thêm vào map id -> name
            self.save_local() # Lưu lại database
            print(f"Đã thêm thành công {name} vào database với {len(embeddings)} ảnh")
        else:
            print(f"{id} đã tồn tại trong db, vui lòng sử dụng hàm cập nhật")

    def re_init(self):
        """ Tái tạo lại database từ dữ liệu đã lưu, sử dụng trong trường hợp database lỗi hoặc tạo lại database
        """
        # Kiểm tra database tồn tại không, nếu có thì xoá
        if os.path.exists(self.path_db):
            os.remove(self.path_db)
        if os.path.exists(self.path_json_id_name):
            os.remove(self.path_json_id_name)
            
        # Load lại database với dữ liệu đã được lưu từ trước
        self.index = load_vt_db(self.path_db)
        self.map_id_name = load_id_name(self.path_json_id_name)
        print("Đã tạo lại db mới")
        
    
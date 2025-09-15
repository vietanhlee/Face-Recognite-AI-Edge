import numpy as np
import faiss
import os
import conf
from utils import init_id_name, init_vt_db, delete_id_name, check_is_id_exist, add_id_name

class VectorBD:
    """Bao bọc thao tác với FAISS index và map id ↔ tên.

    Args:
        path_db: Đường dẫn tệp FAISS index.
        path_json_id_name: Đường dẫn tệp JSON lưu ánh xạ id ↔ tên.
    """
    def __init__(self, path_db : str = conf.path_vector_db, path_json_id_name: str = conf.path_json_id_name):
        self.dim = conf.dim
        self.path_db = path_db
        self.path_json_id_name = path_json_id_name
        self.index = init_vt_db(self.path_db)
        self.map_id_name = init_id_name(self.path_json_id_name)

    def search_emb(self, embeddings: np.ndarray):
        """Tìm lân cận gần nhất cho mỗi embedding trong batch.

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
        """ Xoá tất cả các embedding có id tương ứng là id lẫn xoá luôn trong map id --> name

        Args:
            id (int): id của embedding cần xoá
        """
        # FAISS yêu cầu danh sách id dạng mảng
        self.index.remove_ids(np.array([id]))
        # Lưu index sau khi xoá để đảm bảo dữ liệu nhất quán
        self.save_local()
        # Xoá luôn trong ánh xạ id ↔ tên
        delete_id_name(id, self.path_json_id_name)

    def update_emb(self, embeddings: np.ndarray, id: int):
        """Cập nhật embedding theo id.

        Args:
            embeddings (np.ndarray): Các vector embedding (đingj dạng batch)
            id (int): id gắn với vector embeddings đó
        """
        self.remove_emb(id)
        self.add_emb(embeddings, id)
        print("Đã cập nhật thành công")
        
    def save_local(self):
        """Lưu FAISS index xuống tệp ``self.path_db``."""
        faiss.write_index(self.index, self.path_db)
    
    def add_emb(self, embeddings: np.ndarray, name: str, id: int):
        """Thêm embedding (dạng batch) vào index và cập nhật map id ↔ tên.

        Args:
            embeddings (np.ndarray): Batch embedding (shape (N, dim)).
            name (str): Tên hiển thị của đối tượng.
            id (int): Định danh tương ứng.
        """
        # Kiểm tra id đã tồn tại hay chưa
        if not check_is_id_exist(id, self.path_json_id_name):
            # Thêm vào index embeddings với id tương ứng
            self.index.add_with_ids(embeddings, np.array([id] * len(embeddings)))
            add_id_name(id, name) # Thêm vào map id -> name
            self.save_local() # Lưu lại database
            print(f"Đã thêm thành công {name.split('_')[0]} với ID: {id} vào database với {len(embeddings)} ảnh")
        else:
            print(f"{id} đã tồn tại trong db, vui lòng sử dụng hàm cập nhật")

    def re_init(self):
        """Tạo mới hoàn toàn index và map id ↔ tên trên đĩa.

        Dùng khi cần làm sạch/cài đặt lại cơ sở dữ liệu cục bộ.
        """
        # Kiểm tra database tồn tại không, nếu có thì xoá
        if os.path.exists(self.path_db):
            os.remove(self.path_db)
        if os.path.exists(self.path_json_id_name):
            os.remove(self.path_json_id_name)
            
        # Load lại database với dữ liệu đã được lưu từ trước
        self.index = init_vt_db(self.path_db)
        self.map_id_name = init_id_name(self.path_json_id_name)
        print("Đã tạo lại db mới")
        
    
"""
Tiện ích cập nhật thông tin khuôn mặt trong cơ sở dữ liệu.

Kịch bản:
- Xoá toàn bộ embedding của một ID hiện có
- Đăng ký lại khuôn mặt (thêm embedding mới) cho cùng ID với tên mới

Sử dụng trực tiếp bằng CLI:
    python update_face.py --id 123 --name "Nguyen Van A"
"""

import delete_face
import register_face
import argparse

def main(id: int, name: str):
    """Cập nhật embeddings theo ID bằng cách xoá rồi thêm lại.

    Args:
        id (int): Định danh cần cập nhật.
        name (str): Tên hiển thị tương ứng với ID.
    """
    # Xoá theo ID
    delete_face.main(id = id)
    
    # Thêm gương mặt mới
    register_face.main(name= name, id= id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type= int, required=True, help="ID cần cập nhật")
    parser.add_argument("--name", type= str, required=True, help="Tên tương ứng với ID cần cập nhật")
    args = parser.parse_args()

    main(id= args.id, name= args.name)
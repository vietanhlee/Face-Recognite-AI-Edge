## Hệ thống nhận diện khuôn mặt

### Giới thiệu
Hệ thống nhận diện khuôn mặt thời gian thực sử dụng:
- **YOLO (OpenVINO INT8)** để phát hiện khuôn mặt nhanh và chính xác.
- **Facenet (TFLite INT8)** để trích xuất embedding (128D) và tìm kiếm trong FAISS.

Phù hợp demo, điểm danh, kiểm soát ra vào, có thể chạy trên máy cấu hình vừa phải.

### Chức năng chính
- **Nhận diện realtime** từ webcam (Streamlit Web + streamlit-webrtc).
- **Đăng ký** gương mặt mới (chụp nhiều hướng, lưu embedding).
- **Cập nhật** (xoá embeddings cũ của ID và đăng ký lại).
- **Xoá** theo ID khỏi DB và thư mục ảnh.
- **Khởi tạo/Thêm DB** từ thư mục `images/{id}_{name}/...`.

---

## Cài đặt

### Yêu cầu
- Python >= 3.10

### Cài phụ thuộc
```bash
pip install -r requirements.txt
```

Lưu ý: Trên Windows, nếu gặp lỗi camera với trình duyệt, dùng Chrome mới nhất và cho phép quyền camera.

---

## Chạy demo web (Streamlit)

```bash
streamlit run app.py
```

- Tab "Nhận diện": cấp quyền camera, đưa mặt vào khung để xem kết quả.
- Tab "Đăng ký": nhập Tên và ID, quá trình sẽ mở vòng lặp chụp như CLI (nhấn P để chụp theo hướng, Q để thoát). Ảnh và embeddings sẽ được lưu.
- Tab "Xoá": nhập ID để xoá.
- Tab "Cập nhật": nhập ID và Tên mới, hệ thống sẽ xoá embeddings cũ và đăng ký lại.
- Tab "Khởi tạo DB": chọn thư mục ảnh gốc (mặc định `./images`), chọn "Khởi tạo lại" nếu muốn xoá DB cũ rồi xây lại.

---

## Dùng qua dòng lệnh (CLI)

- Nhận diện realtime (OpenCV):
```bash
python main.py
```

- Đăng ký gương mặt:
```bash
python register_face.py --name "Alice" --id 11
```

- Xoá theo ID:
```bash
python delete_face.py --id 11
```

- Tạo/Thêm DB từ thư mục ảnh (`--reinit 1` để xoá DB cũ và tạo mới; `0` để thêm):
```bash
python create_vt_db.py --reinit 1
```

---

## Cấu trúc thư mục ảnh

```
images/
  11_Alice/
    mid.jpg
    left.jpg
    right.jpg
    up.jpg
    down.jpg
    up_left.jpg
    up_right.jpg
    down_left.jpg
    down_right.jpg
```

---

## Tài liệu tham khảo

- [YOLO](https://github.com/ultralytics/ultralytics)
- [FaceNet](https://github.com/davidsandberg/facenet)
- [OpenVINO](https://docs.openvino.ai/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)

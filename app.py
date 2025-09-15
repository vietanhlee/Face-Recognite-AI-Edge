import os
import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

from FaceRecognite import Regconizer
from VectorDB import VectorBD
from utils import check_is_id_exist, read_image
import create_vt_db
import register_face
import delete_face
import update_face


st.set_page_config(page_title="Face Recognition Demo", layout="wide")


@st.cache_resource(show_spinner=False)
def get_regconizer():
    return Regconizer()


@st.cache_resource(show_spinner=False)
def get_vector_db():
    return VectorBD()


def page_recognize():
    st.subheader("Nhận diện gương mặt (Webcam)")

    reg = get_regconizer()

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    st.info("Cho phép truy cập webcam. Hướng khuôn mặt vào khung để nhận diện.")

    # Lazy import to avoid hard dependency if user runs other tabs only
    import av  # noqa: WPS433

    class VideoProcessor:
        def __init__(self):
            self._reg = reg

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Flip horizontally for a natural mirror view
            img = cv2.flip(img, 1)
            _ = self._reg.regcognize_face(img)
            vis = self._reg.img_with_bbs if self._reg.img_with_bbs is not None else img
            # Downscale to half size
            vis = cv2.resize(vis, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            return av.VideoFrame.from_ndarray(vis, format="bgr24")

    webrtc_streamer(
        key="recognize",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 800},
                "height": {"ideal": 600},
            },
            "audio": False,
        },
        rtc_configuration=rtc_config,
        video_processor_factory=VideoProcessor,
        video_html_attrs={
            "style": {"width": "800px", "height": "600px"},
            "controls": False,
            "autoPlay": True,   
        },
    )


def page_register():
    st.subheader("Đăng ký gương mặt")
    with st.form("register_form"):
        name = st.text_input("Tên", "User")
        id_val = st.number_input("ID", min_value=0, step=1)
        submitted = st.form_submit_button("Bắt đầu đăng ký")

    if submitted:
        if check_is_id_exist(int(id_val)):
            st.error(f"ID {int(id_val)} đã tồn tại. Hãy chọn ID khác hoặc dùng 'Cập nhật'.")
            return
        st.success("Mở giao diện đăng ký trong cửa sổ mới/terminal.")
        st.info("Trong cửa sổ chạy, nhấn P để chụp theo hướng, Q để thoát.")
        # Trigger the existing CLI flow synchronously
        register_face.main(name=str(name), id=int(id_val))


def page_delete():
    st.subheader("Xoá theo ID")
    with st.form("delete_form"):
        id_val = st.number_input("ID", min_value=0, step=1)
        submitted = st.form_submit_button("Xoá")
    if submitted:
        delete_face.main(int(id_val))
        st.success(f"Đã xử lý xoá ID {int(id_val)} (nếu tồn tại).")


def page_update():
    st.subheader("Cập nhật (xoá rồi đăng ký lại)")
    with st.form("update_form"):
        id_val = st.number_input("ID", min_value=0, step=1, key="upd_id")
        name = st.text_input("Tên mới", key="upd_name")
        submitted = st.form_submit_button("Cập nhật")
    if submitted:
        update_face.main(id=int(id_val), name=str(name))
        st.success("Đã cập nhật. Nếu chưa tồn tại, quá trình sẽ đăng ký lại.")


def page_db():
    st.subheader("Khởi tạo/Tải thêm database từ thư mục ảnh")
    root = st.text_input("Thư mục ảnh gốc (images)", value="./images")
    reinit = st.checkbox("Khởi tạo lại hoàn toàn (xoá DB cũ)", value=False)
    if st.button("Thực thi"):
        if not os.path.isdir(root):
            st.error(f"Không tìm thấy thư mục: {root}")
            return
        create_vt_db.add_emb_in_folder(root_folder=root, is_reinit=bool(reinit))
        st.success("Đã hoàn tất xử lý ảnh vào DB.")


def main():
    st.title("Face Recognition Demo")
    st.sidebar.title("Chức năng")
    page = st.sidebar.radio(
        label="Chọn trang",
        options=["Nhận diện", "Đăng ký", "Xoá", "Cập nhật", "Khởi tạo DB"],
    )

    if page == "Nhận diện":
        page_recognize()
    elif page == "Đăng ký":
        page_register()
    elif page == "Xoá":
        page_delete()
    elif page == "Cập nhật":
        page_update()
    else:
        page_db()


if __name__ == "__main__":
    main()



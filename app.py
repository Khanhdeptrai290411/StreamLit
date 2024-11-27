import streamlit as st
import cv2
import time
import base64

with open("static/logo.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Hear Me",
    page_icon=f"data:image/png;base64,{encoded_string}",
    layout="wide"
)

# Custom CSS cho giao diện
st.markdown("""
    <style>
        .navbar {
            background-color: #2563eb;
            padding: 15px;
            color: white;
            font-size: 20px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .navbar img {
            height: 50px;
            width: 50px;
            margin-right: 10px;
            border-radius: 50%; /* Làm tròn logo */
        }
        .stAlert {
            opacity: 1 !important; /* Loại bỏ hiệu ứng mờ */
        }
        .navbar a {
            color: white;
            margin: 0 15px;
            text-decoration: none;
        }
        .navbar a:hover {
            text-decoration: underline;
        }
        .section {
            margin: 20px 0;
        }
        .camera-section {
            border: 2px dashed #90A4AE; /* Màu viền camera */
            background-color: #F0F8FF; /* Màu nền nhạt hơn cho khung camera */
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #546E7A; /* Màu chữ đồng nhất */
        }
        .output-section {
            background-color: #2563eb;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Navbar
st.markdown(f"""
    <div class="navbar">
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{encoded_string}" alt="Logo">
            <span>Hear Me</span>
        </div>
        <div>
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Contact</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div style="text-align: center; background-color: #E3F2FD; padding: 30px; margin: 20px 0; border-radius: 10px;">
        <h1 style="color: #4b71ff;">Sign Language Translator</h1>
        <p style="font-size: 18px; color: #546E7A;">Breaking barriers in communication</p>
    </div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["Sign to Language", "Language to Sign"])

# Tab 1: Input dạng video
with tab1:
    # Checkbox to toggle camera
    toggle_camera = st.checkbox("Turn Camera ON", key="camera_toggle")

    if not toggle_camera:
        # Hiển thị phần "Live Camera Input" khi camera tắt
        st.markdown("""
        <div class="camera-section">
            <h3>Live Camera Input</h3>
            <p>Toggle the button below to turn the camera on/off.</p>
        </div>
        """, unsafe_allow_html=True)

    FRAME_WINDOW = st.empty()  # Khung hiển thị camera

    if toggle_camera:
        cap = cv2.VideoCapture(0)  # Bật camera
        if not cap.isOpened():
            st.error("Cannot access camera. Please check your camera connection.")
        else:
            st.success("Camera is ON!")

            # Vòng lặp hiển thị video
            while toggle_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from camera.")
                    break

                # Chuyển đổi khung hình từ BGR sang RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)  # Hiển thị khung hình

                # Dừng tạm để giảm tải CPU
                time.sleep(0.03)

                # Cập nhật trạng thái của checkbox (thoát nếu camera tắt)
                if not st.session_state.camera_toggle:
                    break

            cap.release()  # Giải phóng camera
            FRAME_WINDOW.empty()  # Xóa khung hình khi camera tắt

# Tab 2: Input dạng text
with tab2:
    st.markdown("""
        <div style="border: 2px dashed #B0BEC5; padding: 50px; text-align: center; color: #4b71ff; border-radius: 10px;">
            <p style="font-size: 16px; color: #546E7A;">Type or paste your text to see the sign translation</p>
            <i style="font-size: 50px; color: #90A4AE;">✏️</i>
        </div>
    """, unsafe_allow_html=True)

# Output Section
st.markdown("""
    <div class="section">
        <h3>Translated Text</h3>
    </div>
""", unsafe_allow_html=True)

# Output hiển thị văn bản từ model
output_text = "Translation will appear here..."  # Placeholder cho đầu ra
st.markdown(f"""
    <div class="output-section">
        <p style="font-size: 16px;">{output_text}</p>
    </div>
""", unsafe_allow_html=True)

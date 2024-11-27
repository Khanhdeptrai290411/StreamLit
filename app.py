import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import torch
from pathlib import Path
import src.configs as cf
import base64
import time
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

yolo_weights_path = str(Path('./yolov5/runs/train/exp/weights/best.pt'))
keras_model_path = str(Path('./model/fine_tune_asl_model.h5'))

with open("static/logo.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Hear Me",
    page_icon=f"data:image/png;base64,{encoded_string}",
    layout="wide"
)
try:
    yolo_model = torch.hub.load(
        './yolov5',
        'custom',
        path=yolo_weights_path,
        source='local',
        force_reload=True
    )
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    st.error(f"Error loading YOLOv5 model: {e}")
    st.stop()

try:
    sign_model = load_model(keras_model_path)
    print("Keras model loaded successfully.")
except Exception as e:
    st.error(f"Error loading Keras model: {e}")
    st.stop()
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
            background-color: #d3d3d3; /* Màu nền xám nhạt */
            color: black; /* Màu chữ đen */
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
# Tab 1: Input dạng video
# Tab 1: Input dạng video
# Tab 1: Input dạng video
with tab1:
    # Checkbox bật/tắt camera
    toggle_camera = st.checkbox("Turn Camera ON", key="camera_toggle")
    
    # Thêm khung hiển thị camera ở ngay dưới checkbox
    FRAME_WINDOW = st.empty()  # Khung hiển thị camera

    # Thêm khung hiển thị văn bản "Translated Text" bên dưới khung camera
    translated_text_placeholder = st.empty()  # Khung dành cho "Translated Text"

    if toggle_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access camera.")
        else:
            word = ""  # Chuỗi để lưu kết quả nhận diện
            padding = 80

            while toggle_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from camera.")
                    break

                frame = cv2.flip(frame, 1)  # Lật ngang hình ảnh

                # Nhận diện bằng YOLOv5
                try:
                    results = yolo_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    detections = results.pandas().xyxy[0]
                except Exception as e:
                    st.error(f"YOLO detection error: {e}")
                    break

                for _, row in detections.iterrows():
                    if row['name'] == 'hand' and row['confidence'] > 0.5:
                        xmin = max(0, int(row['xmin']) - padding)
                        ymin = max(0, int(row['ymin']) - padding)
                        xmax = min(frame.shape[1], int(row['xmax']) + padding)
                        ymax = min(frame.shape[0], int(row['ymax']) + padding)

                        cropped_hand = frame[ymin:ymax, xmin:xmax]

                        # Nhận diện bằng model Keras
                        try:
                            resized_frame = cv2.resize(cropped_hand, (cf.IMAGE_SIZE, cf.IMAGE_SIZE))
                            reshaped_frame = np.array(resized_frame).reshape((1, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 3))
                            frame_for_model = reshaped_frame / 255.0
                            prediction = sign_model.predict(frame_for_model)
                            text = cf.CLASSES[prediction.argmax()]
                            if text != "nothing":  # Loại bỏ các kết quả không hợp lệ
                                word += text  # Thêm chữ vào kết quả
                        except Exception as e:
                            st.error(f"Keras prediction error: {e}")
                            continue

                        # Vẽ bounding box
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(frame, f"{text}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Hiển thị khung hình camera
                FRAME_WINDOW.image(frame, channels="BGR")

                # Cập nhật nội dung hiển thị trong khung "Translated Text"
                translated_text_placeholder.markdown(f"""
                     <div class="section">
                        <h3>Translated Text</h3>
                     </div>                                 
                    <div class="output-section">
                        <p style="font-size: 16px;">{word}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Tạm dừng để giảm tải CPU
                time.sleep(0.03)

            cap.release()
            FRAME_WINDOW.empty()
    else:
        # Nếu camera tắt, đặt khung "Live Camera Input" và nội dung mặc định
        FRAME_WINDOW.markdown("""
         <div class="camera-section">
            <h3>Live Camera Input</h3>
            <p>Toggle the button below to turn the camera on/off.</p>
        </div>
        """, unsafe_allow_html=True)

        # Đặt nội dung mặc định cho "Translated Text"
        translated_text_placeholder.markdown("""
               <div class="section">
                  <h3>Translated Text</h3>
               </div>
            <div class="output-section">
                <p style="font-size: 16px;">Translation will appear here...</p>
            </div>
        """, unsafe_allow_html=True)


    # **Đặt tiêu đề phía trên khung dịch**
    





# Tab 2: Input dạng text
with tab2:
    st.markdown("""
        <div style="border: 2px dashed #B0BEC5; padding: 50px; text-align: center; color: #4b71ff; border-radius: 10px;">
            <p style="font-size: 16px; color: #546E7A;">Type or paste your text to see the sign translation</p>
            <i style="font-size: 50px; color: #90A4AE;">✏️</i>
        </div>
    """, unsafe_allow_html=True)



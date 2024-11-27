import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import torch
from pathlib import Path
import src.configs as cf
import time
import base64
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

# Đường dẫn mô hình
yolo_weights_path = str(Path('./yolov5/runs/train/exp/weights/best.pt'))
keras_model_path = str(Path('./model/fine_tune_asl_model.h5'))

# Tải mô hình YOLOv5
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
    print(f"Error loading YOLOv5 model: {e}")
    exit(1)

# Tải mô hình nhận diện ký hiệu
try:
    sign_model = load_model(keras_model_path)
    print("Keras model loaded successfully.")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    exit(1)

# Thiết lập giao diện Streamlit
st.set_page_config(page_title="Sign Language Translator", layout="wide")
st.title("Sign Language Translator")

FRAME_WINDOW = st.image([])  # Khung hiển thị video
toggle_camera = st.checkbox("Turn Camera ON")  # Nút bật camera

# Hàm nhận diện
def recognize():
    # Mở camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cam.isOpened():
        st.error("Error: Could not open camera.")
        return

    text, word = "", ""
    count_same_frame = 0
    padding = 80

    while toggle_camera:
        ret, frame = cam.read()
        if not ret:
            st.error("Error: Could not read frame from camera.")
            break

        try:
            # YOLOv5 nhận diện
            results = yolo_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = results.pandas().xyxy[0]
        except Exception as e:
            st.error(f"Error during YOLOv5 detection: {e}")
            break

        for _, row in detections.iterrows():
            if row['name'] == 'hand' and row['confidence'] > 0.5:
                xmin = max(0, int(row['xmin']) - padding)
                ymin = max(0, int(row['ymin']) - padding)
                xmax = min(frame.shape[1], int(row['xmax']) + padding)
                ymax = min(frame.shape[0], int(row['ymax']) + padding)

                cropped_hand = frame[ymin:ymax, xmin:xmax]

                try:
                    # Xử lý hình ảnh cho mô hình Keras
                    resized_frame = cv2.resize(cropped_hand, (cf.IMAGE_SIZE, cf.IMAGE_SIZE))
                    reshaped_frame = np.array(resized_frame).reshape((1, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 3))
                    frame_for_model = reshaped_frame / 255.0

                    old_text = text
                    prediction = sign_model.predict(frame_for_model)
                    prediction_probability = prediction[0, prediction.argmax()]
                    text = cf.CLASSES[prediction.argmax()]
                except Exception as e:
                    st.error(f"Error during Keras model prediction: {e}")
                    continue

                if text == 'space':
                    text = '_'
                if text != 'nothing':
                    if old_text == text:
                        count_same_frame += 1
                    else:
                        count_same_frame = 0

                    if count_same_frame > 10:
                        word += text
                        count_same_frame = 0

                if prediction_probability > 0.5:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"{text} ({prediction_probability * 100:.2f}%)",
                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Hiển thị khung hình trên Streamlit
        FRAME_WINDOW.image(frame, channels="BGR")

        # Dừng tạm thời để giảm tải CPU
        time.sleep(0.03)

    cam.release()

# Gọi hàm nhận diện khi bật camera
if toggle_camera:
    recognize()
else:
    st.warning("Camera is OFF. Turn it on using the checkbox above.")

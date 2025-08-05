import cv2
import numpy as np
import streamlit as st
from app.emotion_detector import detect_emotion
from app.llm_response_generator import generate_response

def run_ui():
    st.title("Facial Emotion Recognition with LLM 💬")

    picture = st.camera_input("Take a picture 📸")

    if picture:
        file_bytes = picture.getvalue()
        img_array = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        emotion = detect_emotion(frame)
        response = generate_response(emotion)

        st.image(frame, channels="BGR", caption=f"Detected Emotion: {emotion}")
        st.markdown(f"### 🧠 AI Response: {response}")

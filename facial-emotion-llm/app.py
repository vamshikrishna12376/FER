import streamlit as st
import cv2
from emotion_detector import detect_emotion
from llm_response_generator import generate_response

st.title("Facial Emotion Recognition with LLM")

picture = st.camera_input("Take a picture")

if picture:
    file_bytes = picture.getvalue()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    emotion = detect_emotion(img)
    response = generate_response(emotion)

    st.image(img, channels="BGR", caption=f"Detected Emotion: {emotion}")
    st.markdown(f"**LLM Response:** {response}")

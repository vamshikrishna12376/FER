import cv2
from emotion_detector import detect_emotion

def capture_emotion():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        emotion = detect_emotion(frame)
        cv2.putText(frame, f'Emotion: {emotion}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Facial Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

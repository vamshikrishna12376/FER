import cv2
import numpy as np
from tensorflow.keras.models import load_model
from transformers import pipeline

# Load your pretrained CNN model for facial emotion recognition
cnn_model = load_model('model/emotion_cnn.h5')
emotion_labels = ['anger', 'fear', 'joy', 'sad', 'surprise', 'neutral']

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load LLM text classification pipeline for emotion detection
llm_emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"  # You can replace this with another model if you prefer
)

def predict_facial_emotion(face_img):
    """Predict emotion from a face image using CNN model."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype('float32') / 255.0
    input_arr = np.expand_dims(normalized, axis=(0, -1))
    preds = cnn_model.predict(input_arr)
    emotion_idx = np.argmax(preds)
    return emotion_labels[emotion_idx]

def llm_refine_emotion(text_description):
    """Use LLM to analyze/refine emotion based on descriptive text."""
    result = llm_emotion_classifier(text_description)[0]
    return result['label'].lower(), result['score']

def main():
    cap = cv2.VideoCapture(0)
    print("Starting webcam. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # Get CNN emotion prediction
            cnn_emotion = predict_facial_emotion(face_img)

            # Create a simple descriptive text for LLM
            description = f"The person looks {cnn_emotion}."
            
            # Get refined emotion from LLM based on description
            llm_emotion, confidence = llm_refine_emotion(description)

            # Display results on frame
            label = f"CNN: {cnn_emotion} | LLM: {llm_emotion} ({confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Facial Emotion Recognition with LLM Refinement", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

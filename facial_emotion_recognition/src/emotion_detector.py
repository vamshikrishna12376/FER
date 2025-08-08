"""
Real-time Facial Emotion Recognition using Webcam
Detects faces and predicts emotions in real-time video stream
"""

import cv2
import numpy as np
import os
import sys
from tensorflow import keras
import time

class EmotionDetector:
    def __init__(self, model_path='model/emotion_cnn.h5'):
        self.model_path = model_path
        self.emotion_labels = ['anger', 'fear', 'joy', 'sad', 'surprise', 'neutral']
        self.emotion_colors = {
            'anger': (0, 0, 255),      # Red
            'fear': (128, 0, 128),     # Purple
            'joy': (0, 255, 255),      # Yellow
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 165, 255), # Orange
            'neutral': (0, 255, 0)     # Green
        }
        
        # Load model
        self.model = self.load_model()
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def load_model(self):
        """Load the trained emotion recognition model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found at {self.model_path}")
            print("Please train the model first by running: python src/train_model.py")
            sys.exit(1)
        
        try:
            print(f"📦 Loading model from {self.model_path}...")
            model = keras.models.load_model(self.model_path)
            print("✅ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model prediction"""
        # Resize to model input size
        face_resized = cv2.resize(face_img, (48, 48))
        
        # Normalize pixel values
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Reshape for model input (batch_size, height, width, channels)
        face_input = np.expand_dims(face_normalized, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)
        
        return face_input
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        # Preprocess face
        face_input = self.preprocess_face(face_img)
        
        # Make prediction
        prediction = self.model.predict(face_input, verbose=0)
        
        # Get emotion with highest probability
        emotion_idx = np.argmax(prediction[0])
        emotion = self.emotion_labels[emotion_idx]
        confidence = prediction[0][emotion_idx]
        
        return emotion, confidence
    
    def draw_emotion_info(self, frame, x, y, w, h, emotion, confidence):
        """Draw bounding box and emotion information on frame"""
        # Get color for emotion
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare text
        emotion_text = f"{emotion.capitalize()}"
        confidence_text = f"{confidence:.2f}"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(emotion_text, font, font_scale, thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(confidence_text, font, font_scale-0.2, thickness-1)
        
        # Draw background rectangles for text
        cv2.rectangle(frame, (x, y - text_h - 10), (x + max(text_w, conf_w) + 10, y), color, -1)
        
        # Draw text
        cv2.putText(frame, emotion_text, (x + 5, y - text_h), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, confidence_text, (x + 5, y - 5), font, font_scale-0.2, (255, 255, 255), thickness-1)
    
    def draw_fps(self, frame):
        """Draw FPS counter on frame"""
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self, camera_index=0):
        """Run real-time emotion detection"""
        print("🎥 Starting real-time emotion detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        # Initialize video capture
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        screenshot_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Convert to grayscale for face detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = gray_frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence = self.predict_emotion(face_roi)
                    
                    # Draw emotion information
                    self.draw_emotion_info(frame, x, y, w, h, emotion, confidence)
                
                # Update and draw FPS
                self.update_fps()
                self.draw_fps(frame)
                
                # Draw instructions
                cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Facial Emotion Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f'screenshot_{screenshot_counter:03d}.png'
                    cv2.imwrite(screenshot_path, frame)
                    print(f"📸 Screenshot saved as {screenshot_path}")
                    screenshot_counter += 1
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera released and windows closed")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Facial Emotion Recognition')
    parser.add_argument('--model', default='model/emotion_cnn.h5', 
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    
    args = parser.parse_args()
    
    # Create and run detector
    detector = EmotionDetector(model_path=args.model)
    detector.run(camera_index=args.camera)

if __name__ == "__main__":
    main()

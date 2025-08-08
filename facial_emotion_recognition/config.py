"""
Configuration file for Facial Emotion Recognition project
Contains all configurable parameters and settings
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Data configuration
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
FER2013_CSV_PATH = os.path.join(DATA_DIR, 'fer2013.csv')

# Model configuration
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'emotion_cnn.h5')
IMG_SIZE = (48, 48)
NUM_CLASSES = 6

# Emotion labels (consistent across all modules)
EMOTION_LABELS = ['anger', 'fear', 'joy', 'sad', 'surprise', 'neutral']

# Emotion colors for visualization (BGR format for OpenCV)
EMOTION_COLORS = {
    'anger': (0, 0, 255),      # Red
    'fear': (128, 0, 128),     # Purple
    'joy': (0, 255, 255),      # Yellow
    'sad': (255, 0, 0),        # Blue
    'surprise': (0, 165, 255), # Orange
    'neutral': (0, 255, 0)     # Green
}

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Data augmentation parameters
ROTATION_RANGE = 10
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.1

# Camera configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Face detection parameters
FACE_CASCADE_SCALE_FACTOR = 1.1
FACE_CASCADE_MIN_NEIGHBORS = 5
FACE_CASCADE_MIN_SIZE = (30, 30)

# LLM configuration
DEFAULT_TEXT_MODEL = 'j-hartmann/emotion-english-distilroberta-base'
FACE_EMOTION_WEIGHT = 0.6  # Weight for facial emotion in multimodal analysis

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Performance settings
USE_GPU = True  # Set to False to force CPU usage
FPS_UPDATE_INTERVAL = 30  # Update FPS counter every N frames

# File extensions
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(TEST_DATA_DIR, exist_ok=True)
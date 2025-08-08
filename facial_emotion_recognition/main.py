#!/usr/bin/env python3
"""
Facial Emotion Recognition (FER) - Main Application Entry Point
Capstone Project

This is the main entry point for the facial emotion recognition system.
It provides a simple interface to run different components of the project.
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition System')
    parser.add_argument('--mode', choices=['detect', 'train', 'preprocess', 'llm'], 
                       default='detect', help='Mode to run the application in')
    parser.add_argument('--model-path', default='model/emotion_cnn.h5', 
                       help='Path to the trained model')
    parser.add_argument('--data-path', default='data/', 
                       help='Path to the dataset')
    
    args = parser.parse_args()
    
    print("🎭 Facial Emotion Recognition System")
    print("=" * 50)
    
    if args.mode == 'detect':
        print("🎥 Starting real-time emotion detection...")
        try:
            from emotion_detector import EmotionDetector
            detector = EmotionDetector(model_path=args.model_path)
            detector.run()
        except ImportError as e:
            print(f"❌ Error importing emotion detector: {e}")
            print("Make sure all dependencies are installed: pip install -r requirements.txt")
        except Exception as e:
            print(f"❌ Error running emotion detection: {e}")
    
    elif args.mode == 'train':
        print("🧠 Starting model training...")
        try:
            from train_model import train_emotion_model
            train_emotion_model(data_path=args.data_path, model_save_path=args.model_path)
        except ImportError as e:
            print(f"❌ Error importing training module: {e}")
        except Exception as e:
            print(f"❌ Error during training: {e}")
    
    elif args.mode == 'preprocess':
        print("📊 Starting data preprocessing...")
        try:
            from preprocess import preprocess_data
            preprocess_data(data_path=args.data_path)
        except ImportError as e:
            print(f"❌ Error importing preprocessing module: {e}")
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
    
    elif args.mode == 'llm':
        print("🤖 Starting LLM emotion analysis...")
        try:
            from llm_integration import run_llm_emotion_analysis
            run_llm_emotion_analysis()
        except ImportError as e:
            print(f"❌ Error importing LLM module: {e}")
        except Exception as e:
            print(f"❌ Error running LLM analysis: {e}")

if __name__ == "__main__":
    main()
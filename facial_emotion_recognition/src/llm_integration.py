"""
LLM Integration Module for Text-based Emotion Analysis
Complements facial emotion recognition with text emotion analysis
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available. Install with: pip install transformers torch")

class TextEmotionAnalyzer:
    def __init__(self, model_name='j-hartmann/emotion-english-distilroberta-base'):
        """
        Initialize text emotion analyzer
        
        Args:
            model_name: HuggingFace model name for emotion classification
        """
        self.model_name = model_name
        self.emotion_pipeline = None
        self.emotion_mapping = {
            'anger': 'anger',
            'fear': 'fear', 
            'joy': 'joy',
            'sadness': 'sad',
            'surprise': 'surprise',
            'neutral': 'neutral',
            'disgust': 'neutral'  # Map disgust to neutral for consistency
        }
        
        if TRANSFORMERS_AVAILABLE:
            self.load_model()
        else:
            print("❌ Transformers library not available")
    
    def load_model(self):
        """Load the emotion classification model"""
        try:
            print(f"📦 Loading text emotion model: {self.model_name}")
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Text emotion model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Falling back to simple keyword-based analysis...")
            self.emotion_pipeline = None
    
    def analyze_text_simple(self, text: str) -> Tuple[str, float]:
        """Simple keyword-based emotion analysis fallback"""
        emotion_keywords = {
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'irritated'],
            'fear': ['scared', 'afraid', 'terrified', 'anxious', 'worried', 'nervous'],
            'joy': ['happy', 'joyful', 'excited', 'glad', 'cheerful', 'delighted', 'pleased'],
            'sad': ['sad', 'depressed', 'unhappy', 'miserable', 'gloomy', 'down'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'neutral': ['okay', 'fine', 'normal', 'regular', 'usual']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        
        if max(emotion_scores.values()) == 0:
            return 'neutral', 0.5
        
        predicted_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[predicted_emotion] / len(text.split())
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        return predicted_emotion, confidence
    
    def analyze_text(self, text: str) -> Tuple[str, float]:
        """
        Analyze emotion from text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (emotion, confidence)
        """
        if not text.strip():
            return 'neutral', 0.0
        
        if self.emotion_pipeline is None:
            return self.analyze_text_simple(text)
        
        try:
            # Get prediction from transformer model
            results = self.emotion_pipeline(text)
            
            # Extract emotion and confidence
            predicted_label = results[0]['label'].lower()
            confidence = results[0]['score']
            
            # Map to our emotion labels
            emotion = self.emotion_mapping.get(predicted_label, 'neutral')
            
            return emotion, confidence
            
        except Exception as e:
            print(f"⚠️  Error in transformer analysis: {e}")
            return self.analyze_text_simple(text)
    
    def batch_analyze(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Analyze multiple texts at once"""
        results = []
        for text in texts:
            emotion, confidence = self.analyze_text(text)
            results.append((emotion, confidence))
        return results

class MultimodalEmotionAnalyzer:
    """Combines facial and text emotion analysis"""
    
    def __init__(self):
        self.text_analyzer = TextEmotionAnalyzer()
        self.emotion_labels = ['anger', 'fear', 'joy', 'sad', 'surprise', 'neutral']
    
    def combine_emotions(self, face_emotion: str, face_confidence: float,
                        text_emotion: str, text_confidence: float,
                        face_weight: float = 0.6) -> Tuple[str, float]:
        """
        Combine facial and text emotion predictions
        
        Args:
            face_emotion: Predicted emotion from face
            face_confidence: Confidence of face prediction
            text_emotion: Predicted emotion from text
            text_confidence: Confidence of text prediction
            face_weight: Weight for facial emotion (0-1)
            
        Returns:
            Combined emotion and confidence
        """
        text_weight = 1.0 - face_weight
        
        # If emotions match, boost confidence
        if face_emotion == text_emotion:
            combined_confidence = min(1.0, face_confidence * face_weight + text_confidence * text_weight + 0.2)
            return face_emotion, combined_confidence
        
        # If emotions differ, use weighted average approach
        face_score = face_confidence * face_weight
        text_score = text_confidence * text_weight
        
        if face_score > text_score:
            return face_emotion, face_score
        else:
            return text_emotion, text_score
    
    def analyze_multimodal(self, face_emotion: str, face_confidence: float, 
                          text: str) -> Dict[str, any]:
        """
        Perform multimodal emotion analysis
        
        Returns:
            Dictionary with analysis results
        """
        # Analyze text emotion
        text_emotion, text_confidence = self.text_analyzer.analyze_text(text)
        
        # Combine emotions
        combined_emotion, combined_confidence = self.combine_emotions(
            face_emotion, face_confidence, text_emotion, text_confidence
        )
        
        return {
            'face_emotion': face_emotion,
            'face_confidence': face_confidence,
            'text_emotion': text_emotion,
            'text_confidence': text_confidence,
            'combined_emotion': combined_emotion,
            'combined_confidence': combined_confidence,
            'text_input': text
        }

def interactive_text_analysis():
    """Interactive text emotion analysis"""
    print("🤖 Text Emotion Analysis")
    print("=" * 40)
    print("Enter text to analyze emotions (type 'quit' to exit)")
    
    analyzer = TextEmotionAnalyzer()
    
    while True:
        text = input("\n📝 Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        emotion, confidence = analyzer.analyze_text(text)
        
        print(f"😊 Predicted Emotion: {emotion.capitalize()}")
        print(f"🎯 Confidence: {confidence:.2f}")
        
        # Show emotion emoji
        emotion_emojis = {
            'anger': '😠',
            'fear': '😨', 
            'joy': '😃',
            'sad': '😢',
            'surprise': '😲',
            'neutral': '😐'
        }
        emoji = emotion_emojis.get(emotion, '🤔')
        print(f"   {emoji} {emotion.capitalize()}")

def demo_multimodal_analysis():
    """Demo of multimodal emotion analysis"""
    print("🎭 Multimodal Emotion Analysis Demo")
    print("=" * 40)
    
    analyzer = MultimodalEmotionAnalyzer()
    
    # Sample scenarios
    scenarios = [
        {
            'face_emotion': 'joy',
            'face_confidence': 0.85,
            'text': "I'm so happy today! Everything is going great!"
        },
        {
            'face_emotion': 'neutral',
            'face_confidence': 0.60,
            'text': "I'm really angry about what happened at work today."
        },
        {
            'face_emotion': 'sad',
            'face_confidence': 0.75,
            'text': "Feeling a bit down but trying to stay positive."
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 Scenario {i}:")
        print(f"Face: {scenario['face_emotion']} ({scenario['face_confidence']:.2f})")
        print(f"Text: '{scenario['text']}'")
        
        result = analyzer.analyze_multimodal(
            scenario['face_emotion'],
            scenario['face_confidence'],
            scenario['text']
        )
        
        print(f"Text Emotion: {result['text_emotion']} ({result['text_confidence']:.2f})")
        print(f"Combined: {result['combined_emotion']} ({result['combined_confidence']:.2f})")

def run_llm_emotion_analysis():
    """Main function for LLM emotion analysis"""
    print("🤖 LLM Emotion Analysis Module")
    print("=" * 50)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not available")
        print("Install with: pip install transformers torch")
        return
    
    while True:
        print("\nChoose an option:")
        print("1. Interactive text analysis")
        print("2. Multimodal analysis demo")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            interactive_text_analysis()
        elif choice == '2':
            demo_multimodal_analysis()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    run_llm_emotion_analysis()
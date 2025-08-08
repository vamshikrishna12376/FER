"""
Setup script for Facial Emotion Recognition project
Handles initial setup and dependency checking
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_virtual_environment():
    """Check if running in virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment")
        print("Recommended: Create and activate a virtual environment")
        print("python -m venv fer_env")
        print("source fer_env/bin/activate  # On Windows: fer_env\\Scripts\\activate")
    return in_venv

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_opencv():
    """Check OpenCV installation and camera access"""
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access: OK")
            cap.release()
        else:
            print("⚠️  Camera access: Failed (camera may not be available)")
        return True
    except ImportError:
        print("❌ OpenCV not installed")
        return False

def check_tensorflow():
    """Check TensorFlow installation"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU devices available: {len(gpus)}")
        else:
            print("ℹ️  No GPU devices found, using CPU")
        return True
    except ImportError:
        print("❌ TensorFlow not installed")
        return False

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'data',
        'data/train',
        'data/test',
        'data/processed',
        'model',
        'logs',
        'screenshots'
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    # Create emotion subdirectories
    emotions = ['anger', 'fear', 'joy', 'sad', 'surprise', 'neutral']
    for split in ['train', 'test']:
        for emotion in emotions:
            emotion_dir = os.path.join('data', split, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
    
    print("✅ Directory structure created")

def check_system_requirements():
    """Check system-specific requirements"""
    system = platform.system()
    print(f"💻 Operating System: {system}")
    
    if system == "Darwin":  # macOS
        print("ℹ️  macOS detected")
        print("   Make sure Xcode Command Line Tools are installed:")
        print("   xcode-select --install")
    elif system == "Linux":
        print("ℹ️  Linux detected")
        print("   Make sure you have the required system packages:")
        print("   sudo apt-get install python3-dev libopencv-dev")
    elif system == "Windows":
        print("ℹ️  Windows detected")
        print("   Make sure Visual C++ redistributables are installed")

def download_sample_data():
    """Provide instructions for downloading sample data"""
    print("\n📊 Dataset Setup:")
    print("=" * 50)
    print("To train the model, you need the FER-2013 dataset:")
    print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Download the fer2013.csv file")
    print("3. Place it in the 'data/' directory")
    print("4. Run: python src/preprocess.py")

def main():
    """Main setup function"""
    print("🎭 Facial Emotion Recognition - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    check_virtual_environment()
    
    # Create directory structure
    create_directory_structure()
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        install_requirements()
    else:
        print("⚠️  requirements.txt not found")
    
    # Check key dependencies
    check_opencv()
    check_tensorflow()
    
    # Check system requirements
    check_system_requirements()
    
    # Dataset instructions
    download_sample_data()
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Download the FER-2013 dataset (see instructions above)")
    print("2. Run preprocessing: python src/preprocess.py")
    print("3. Train the model: python src/train_model.py")
    print("4. Test emotion detection: python main.py")

if __name__ == "__main__":
    main()
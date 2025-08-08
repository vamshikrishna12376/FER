"""
Test script to verify the facial emotion recognition setup
Runs basic tests to ensure all components are working
"""

import os
import sys
import numpy as np
import cv2

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    required_packages = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('tensorflow', 'TensorFlow')
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages imported successfully")
    return True

def test_directory_structure():
    """Test if directory structure exists"""
    print("\n🧪 Testing directory structure...")
    
    required_dirs = [
        'data',
        'data/train',
        'data/test',
        'model',
        'src'
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}")
        else:
            print(f"   ❌ {directory}")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {', '.join(missing_dirs)}")
        print("Run: python setup.py")
        return False
    
    print("✅ Directory structure is correct")
    return True

def test_source_files():
    """Test if source files exist and can be imported"""
    print("\n🧪 Testing source files...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    source_files = [
        ('preprocess', 'src/preprocess.py'),
        ('train_model', 'src/train_model.py'),
        ('emotion_detector', 'src/emotion_detector.py'),
        ('llm_integration', 'src/llm_integration.py')
    ]
    
    failed_imports = []
    
    for module, filepath in source_files:
        if os.path.exists(filepath):
            try:
                __import__(module)
                print(f"   ✅ {filepath}")
            except ImportError as e:
                print(f"   ⚠️  {filepath} (import error: {e})")
                failed_imports.append(filepath)
        else:
            print(f"   ❌ {filepath} (file not found)")
            failed_imports.append(filepath)
    
    if failed_imports:
        print(f"\n⚠️  Some source files have issues: {', '.join(failed_imports)}")
        return False
    
    print("✅ All source files are accessible")
    return True

def test_camera_access():
    """Test camera access"""
    print("\n🧪 Testing camera access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"   ✅ Camera working (frame shape: {frame.shape})")
                cap.release()
                return True
            else:
                print("   ❌ Camera opened but cannot read frames")
                cap.release()
                return False
        else:
            print("   ❌ Cannot open camera")
            return False
    except Exception as e:
        print(f"   ❌ Camera test failed: {e}")
        return False

def test_face_detection():
    """Test face detection cascade"""
    print("\n🧪 Testing face detection...")
    
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if face_cascade.empty():
            print("   ❌ Face cascade classifier is empty")
            return False
        
        # Create a dummy image for testing
        test_image = np.zeros((100, 100), dtype=np.uint8)
        faces = face_cascade.detectMultiScale(test_image)
        
        print("   ✅ Face detection cascade loaded successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Face detection test failed: {e}")
        return False

def test_tensorflow_gpu():
    """Test TensorFlow GPU availability"""
    print("\n🧪 Testing TensorFlow GPU...")
    
    try:
        import tensorflow as tf
        
        print(f"   TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   ✅ GPU devices available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"      GPU {i}: {gpu.name}")
        else:
            print("   ℹ️  No GPU devices found, will use CPU")
        
        # Test basic tensor operations
        with tf.device('/CPU:0'):
            a = tf.constant([1, 2, 3])
            b = tf.constant([4, 5, 6])
            c = tf.add(a, b)
            print(f"   ✅ Basic tensor operations working: {c.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ TensorFlow test failed: {e}")
        return False

def test_model_creation():
    """Test if we can create a simple model"""
    print("\n🧪 Testing model creation...")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Create a simple test model
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(6, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("   ✅ Model creation successful")
        print(f"   Model parameters: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🎭 Facial Emotion Recognition - System Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Directory Structure", test_directory_structure),
        ("Source Files", test_source_files),
        ("Camera Access", test_camera_access),
        ("Face Detection", test_face_detection),
        ("TensorFlow GPU", test_tensorflow_gpu),
        ("Model Creation", test_model_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print("-" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Download FER-2013 dataset to data/fer2013.csv")
        print("2. Run: python src/preprocess.py")
        print("3. Run: python src/train_model.py")
        print("4. Run: python main.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("Run: python setup.py to fix common issues")

if __name__ == "__main__":
    run_all_tests()
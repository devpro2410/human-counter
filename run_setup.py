#!/usr/bin/env python3
"""
Complete setup script for Human Counting System
"""
import os
import sys
import subprocess
import platform

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    directories = [
        "models",
        "uploads", 
        "processed",
        "static/results",
        "input",
        "output"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    return True

def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")
    prototxt_path = "models/MobileNetSSD_deploy.prototxt"
    caffemodel_path = "models/MobileNetSSD_deploy.caffemodel"
    
    prototxt_exists = os.path.exists(prototxt_path)
    caffemodel_exists = os.path.exists(caffemodel_path)
    
    if prototxt_exists:
        print("✅ MobileNetSSD_deploy.prototxt found")
    else:
        print("❌ MobileNetSSD_deploy.prototxt missing")
    
    if caffemodel_exists:
        print("✅ MobileNetSSD_deploy.caffemodel found")
        size = os.path.getsize(caffemodel_path) / (1024*1024)
        print(f"   File size: {size:.1f} MB")
    else:
        print("❌ MobileNetSSD_deploy.caffemodel missing")
    
    return prototxt_exists and caffemodel_exists

def download_models():
    """Download model files"""
    print("\n⬇️  Downloading model files...")
    try:
        from scripts.download_models_gui import download_mobilenet_ssd_models
        return download_mobilenet_ssd_models()
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False

def test_system():
    """Test the system"""
    print("\n🧪 Testing system...")
    try:
        from scripts.main import HumanCounter
        counter = HumanCounter()
        if counter.load_model():
            print("✅ System test passed - Model loaded successfully")
            return True
        else:
            print("❌ System test failed - Could not load model")
            return False
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def show_next_steps():
    """Show next steps to user"""
    print_header("🎉 SETUP COMPLETE!")
    print("\n🚀 Next Steps:")
    print("1. Start the web application:")
    print("   python app.py")
    print("\n2. Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n3. Upload an image or video to test the system")
    print("\n📚 Additional Options:")
    print("• Command line usage: python scripts/main.py --input your_file.jpg")
    print("• Setup page: http://localhost:5000/setup")
    print("• Test with sample files in the input/ directory")

def main():
    """Main setup function"""
    print_header("Human Counting System - Setup")
    print("This script will set up your Human Counting System")
    print("with OpenCV and MobileNet SSD for person detection.")
    
    # Step 1: Check Python version
    print_header("Step 1: System Requirements")
    if not check_python_version():
        return False
    
    # Step 2: Create directories
    print_header("Step 2: Directory Structure")
    create_directories()
    
    # Step 3: Install requirements
    print_header("Step 3: Install Dependencies")
    if not install_requirements():
        print("⚠️  You can try installing manually: pip install -r requirements.txt")
    
    # Step 4: Check/Download model files
    print_header("Step 4: Model Files")
    if not check_model_files():
        print("\n🔄 Attempting to download model files...")
        if not download_models():
            print("\n⚠️  Model files need to be downloaded manually.")
            print("Please visit the setup page after starting the application.")
    
    # Step 5: Test system
    print_header("Step 5: System Test")
    if check_model_files():
        test_system()
    else:
        print("⏭️  Skipping system test - model files not available")
    
    # Show next steps
    show_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user.")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check the error and try again.")
    
    input("\nPress Enter to exit...")

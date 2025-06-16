import os
import urllib.request
from pathlib import Path

def download_mobilenet_ssd():
    """Download MobileNet SSD model files"""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Model URLs (these are example URLs - you may need to find current working links)
    prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    model_url = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
    
    prototxt_path = "models/MobileNetSSD_deploy.prototxt"
    model_path = "models/MobileNetSSD_deploy.caffemodel"
    
    print("Downloading MobileNet SSD model files...")
    print("Note: The caffemodel file is large (~23MB) and may take some time.")
    
    try:
        # Download prototxt file
        if not os.path.exists(prototxt_path):
            print("Downloading prototxt file...")
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
            print(f"Downloaded: {prototxt_path}")
        else:
            print(f"Already exists: {prototxt_path}")
        
        # Note about caffemodel
        print(f"\nFor the caffemodel file ({model_path}), please:")
        print("1. Visit: https://github.com/chuanqi305/MobileNet-SSD")
        print("2. Download MobileNetSSD_deploy.caffemodel")
        print("3. Place it in the models/ directory")
        print("\nAlternatively, you can find it at:")
        print("https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view")
        
    except Exception as e:
        print(f"Error downloading files: {e}")
        print("\nPlease manually download the model files:")
        print("1. MobileNetSSD_deploy.prototxt")
        print("2. MobileNetSSD_deploy.caffemodel")
        print("Place them in the 'models/' directory")

if __name__ == "__main__":
    download_mobilenet_ssd()

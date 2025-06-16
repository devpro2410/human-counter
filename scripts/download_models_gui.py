import os
import requests
from pathlib import Path
import sys

def download_file(url, filename, description):
    """Download a file with progress indication"""
    print(f"\nDownloading {description}...")
    print(f"URL: {url}")
    print(f"Saving to: {filename}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')
                    else:
                        print(f"\rDownloaded: {downloaded} bytes", end='')
        
        print(f"\n✅ Successfully downloaded: {filename}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {description}: {e}")
        return False

def download_mobilenet_ssd_models():
    """Download MobileNet SSD model files"""
    print("🚀 MobileNet SSD Model Downloader")
    print("=" * 50)
    
    # URLs for the model files
    prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    
    # Note: The caffemodel file is large and hosted on Google Drive
    # We'll provide instructions for manual download
    
    models_dir = "models"
    prototxt_path = os.path.join(models_dir, "MobileNetSSD_deploy.prototxt")
    caffemodel_path = os.path.join(models_dir, "MobileNetSSD_deploy.caffemodel")
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Download prototxt file
    success_prototxt = download_file(prototxt_url, prototxt_path, "MobileNet SSD Architecture (.prototxt)")
    
    # Check if caffemodel already exists
    if os.path.exists(caffemodel_path):
        print(f"\n✅ Model weights file already exists: {caffemodel_path}")
        success_caffemodel = True
    else:
        print(f"\n📥 Manual Download Required for Model Weights")
        print("=" * 50)
        print("The MobileNet SSD weights file (.caffemodel) is large (~23MB)")
        print("and needs to be downloaded manually from one of these sources:")
        print()
        print("🔗 Option 1 - GitHub Release:")
        print("   https://github.com/chuanqi305/MobileNet-SSD/blob/master/README.md")
        print()
        print("🔗 Option 2 - Direct Download:")
        print("   https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view")
        print()
        print("🔗 Option 3 - Alternative Mirror:")
        print("   https://github.com/chuanqi305/MobileNet-SSD/tree/master")
        print()
        print("📁 Save the file as:")
        print(f"   {os.path.abspath(caffemodel_path)}")
        print()
        
        # Try to download from a direct link (may not always work)
        caffemodel_urls = [
            "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
            "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
        ]
        
        success_caffemodel = False
        for url in caffemodel_urls:
            print(f"\n🔄 Trying to download from: {url}")
            if download_file(url, caffemodel_path, "MobileNet SSD Weights (.caffemodel)"):
                success_caffemodel = True
                break
        
        if not success_caffemodel:
            print("\n⚠️  Automatic download failed. Please download manually using the links above.")
    
    # Final status
    print("\n" + "=" * 50)
    print("📊 DOWNLOAD STATUS")
    print("=" * 50)
    
    if success_prototxt:
        print("✅ MobileNetSSD_deploy.prototxt - Downloaded")
    else:
        print("❌ MobileNetSSD_deploy.prototxt - Failed")
    
    if success_caffemodel:
        print("✅ MobileNetSSD_deploy.caffemodel - Ready")
    else:
        print("⏳ MobileNetSSD_deploy.caffemodel - Manual download required")
    
    if success_prototxt and success_caffemodel:
        print("\n🎉 All model files are ready!")
        print("You can now run the Human Counting System:")
        print("   python app.py")
    else:
        print("\n⚠️  Setup incomplete. Please download missing files.")
    
    return success_prototxt and success_caffemodel

if __name__ == "__main__":
    try:
        download_mobilenet_ssd_models()
    except KeyboardInterrupt:
        print("\n\n⏹️  Download interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    input("\nPress Enter to exit...")

import requests
import os

def download_file(url, filename):
    print(f"📥 Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r  Progress: {progress:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
        
        print(f"\n✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    os.makedirs("models", exist_ok=True)
    
    files = {
        "models/yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
        "models/coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
        "models/yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    }
    
    for filename, url in files.items():
        if not os.path.exists(filename):
            download_file(url, filename)
        else:
            print(f"✅ {filename} already exists")
    
    print("\n🎉 YOLO model setup complete!")

if __name__ == "__main__":
    main()
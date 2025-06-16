import os
import urllib.request
import cv2
import numpy as np

def create_demo_structure():
    """Create the required folder structure and download demo files"""
    
    # Create directories
    directories = ['models', 'input', 'output']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("\nFolder structure created successfully!")
    print("\nTo use this system, you need to:")
    print("1. Download the MobileNet SSD model files:")
    print("   - MobileNetSSD_deploy.prototxt")
    print("   - MobileNetSSD_deploy.caffemodel")
    print("   Place them in the 'models/' directory")
    print("\n2. Add your input files:")
    print("   - For images: place in 'input/' directory (e.g., input.jpg)")
    print("   - For videos: place in 'input/' directory (e.g., input.mp4)")
    print("\n3. Run the main script:")
    print("   python main.py --input input/your_file.jpg")

def create_sample_image():
    """Create a sample image with some shapes (for testing without real people)"""
    # Create a sample image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img.fill(50)  # Dark gray background
    
    # Add some text
    cv2.putText(img, "Sample Image for Testing", (150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Place your actual image/video in input/ folder", (80, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(img, "Download MobileNet SSD model files to models/ folder", (60, 430), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Save sample image
    cv2.imwrite('input/sample.jpg', img)
    print("Created sample image: input/sample.jpg")

if __name__ == "__main__":
    create_demo_structure()
    create_sample_image()

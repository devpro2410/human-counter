import cv2
import numpy as np
import os
from main import HumanCounter

def create_test_image_with_people():
    """Create a test image with person-like shapes for testing"""
    # Create a test image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img.fill(100)  # Gray background
    
    # Add title
    cv2.putText(img, "Test Image - Human Detection System", (150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw some rectangles to simulate people (for basic testing)
    # These won't be detected as people by the actual model, but help test the pipeline
    people_positions = [
        (100, 150, 180, 350),  # x1, y1, x2, y2
        (250, 200, 320, 400),
        (450, 180, 520, 380),
        (600, 160, 670, 360)
    ]
    
    for i, (x1, y1, x2, y2) in enumerate(people_positions):
        # Draw person-like shape
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 150, 255), 2)
        cv2.putText(img, f"Person {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 1)
    
    # Add instructions
    cv2.putText(img, "Note: This is a test image. For real detection,", (50, 500), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(img, "use actual photos/videos with people.", (50, 530), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Save test image
    os.makedirs('input', exist_ok=True)
    cv2.imwrite('input/test_image.jpg', img)
    print("Created test image: input/test_image.jpg")
    return 'input/test_image.jpg'

def test_system():
    """Test the human counting system"""
    print("Testing Human Counting System...")
    
    # Create test image
    test_image_path = create_test_image_with_people()
    
    # Check if model files exist
    prototxt_path = "models/MobileNetSSD_deploy.prototxt"
    model_path = "models/MobileNetSSD_deploy.caffemodel"
    
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print("\n❌ Model files not found!")
        print("Please download the MobileNet SSD model files:")
        print("1. MobileNetSSD_deploy.prototxt")
        print("2. MobileNetSSD_deploy.caffemodel")
        print("Place them in the 'models/' directory")
        print("\nYou can run: python download_models.py")
        return False
    
    # Test the system
    try:
        counter = HumanCounter()
        print(f"\n✅ Testing with: {test_image_path}")
        counter.run(test_image_path, 'output/test_result.jpg', show_result=True)
        print("✅ Test completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("="*60)
    print("\n1. Process an image:")
    print("   python main.py --input input/your_image.jpg")
    print("\n2. Process a video:")
    print("   python main.py --input input/your_video.mp4")
    print("\n3. Process with custom output:")
    print("   python main.py --input input/image.jpg --output output/my_result.jpg")
    print("\n4. Process without displaying result:")
    print("   python main.py --input input/video.mp4 --no-display")
    print("\n5. Use default input file:")
    print("   python main.py")
    print("   (looks for input/input.jpg by default)")

if __name__ == "__main__":
    # Run setup first
    from setup_demo import create_demo_structure
    create_demo_structure()
    
    # Test the system
    success = test_system()
    
    # Show usage examples
    show_usage_examples()
    
    if success:
        print("\n🎉 System is ready to use!")
    else:
        print("\n⚠️  Please download model files before using the system.")

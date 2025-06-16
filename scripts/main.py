import cv2
import numpy as np
import os
import argparse
from pathlib import Path

class HumanCounter:
    def __init__(self):
        # Initialize the MobileNet SSD model
        self.net = None
        self.classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
        self.person_class_id = 15  # 'person' class in COCO dataset
        self.confidence_threshold = 0.4
        
    def load_model(self):
        """Load the MobileNet SSD model"""
        try:
            prototxt_path = "models/MobileNetSSD_deploy.prototxt"
            model_path = "models/MobileNetSSD_deploy.caffemodel"
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Check if model files exist
            if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
                print("Model files not found. Please download:")
                print("1. MobileNetSSD_deploy.prototxt")
                print("2. MobileNetSSD_deploy.caffemodel")
                print("Place them in the 'models/' directory")
                return False
                
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect_people(self, frame):
        """Detect people in a frame and return bounding boxes"""
        height, width = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame, 0.007843, (300, 300), 127.5
        )
        
        # Set input to the network
        self.net.setInput(blob)
        
        # Run forward pass
        detections = self.net.forward()
        
        people_boxes = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            
            # Filter for person class with sufficient confidence
            if class_id == self.person_class_id and confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x1, y1, x2, y2 = box.astype(int)
                
                people_boxes.append((x1, y1, x2, y2, confidence))
        
        return people_boxes
    
    def draw_detections(self, frame, people_boxes):
        """Draw bounding boxes and count on frame"""
        people_count = len(people_boxes)
        
        # Draw bounding boxes
        for (x1, y1, x2, y2, confidence) in people_boxes:
            # Draw red bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Optional: Draw confidence score
            # label = f"Person: {confidence:.2f}"
            # cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw people count
        count_text = f"Total no. of people : {people_count}"
        cv2.putText(frame, count_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, people_count
    
    def process_image(self, input_path, output_path=None, show_result=True):
        """Process a single image"""
        # Read image
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Could not read image from {input_path}")
            return
        
        print(f"Processing image: {input_path}")
        
        # Detect people
        people_boxes = self.detect_people(frame)
        
        # Draw detections
        result_frame, people_count = self.draw_detections(frame, people_boxes)
        
        print(f"Detected {people_count} people in the image")
        
        # Save result if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, result_frame)
            print(f"Result saved to: {output_path}")
        
        # Show result
        if show_result:
            cv2.imshow("Human Detection Result", result_frame)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def process_video(self, input_path, output_path=None, show_result=True):
        """Process a video file"""
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video from {input_path}")
            return
        
        print(f"Processing video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect people
            people_boxes = self.detect_people(frame)
            
            # Draw detections
            result_frame, people_count = self.draw_detections(frame, people_boxes)
            
            # Write frame to output video
            if out:
                out.write(result_frame)
            
            # Show result
            if show_result:
                cv2.imshow("Human Detection - Video", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing interrupted by user")
                    break
            
            # Print progress
            if frame_count % 30 == 0:  # Print every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_count}/{total_frames} - People: {people_count}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
            print(f"Result video saved to: {output_path}")
        
        if show_result:
            cv2.destroyAllWindows()
        
        print(f"Video processing completed! Processed {frame_count} frames")
    
    def detect_input_type(self, input_path):
        """Detect if input is image or video based on extension"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        ext = Path(input_path).suffix.lower()
        
        if ext in image_extensions:
            return 'image'
        elif ext in video_extensions:
            return 'video'
        else:
            return 'unknown'
    
    def run(self, input_path, output_path=None, show_result=True):
        """Main processing function"""
        # Load model
        if not self.load_model():
            return
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            return
        
        # Detect input type
        input_type = self.detect_input_type(input_path)
        
        if input_type == 'image':
            self.process_image(input_path, output_path, show_result)
        elif input_type == 'video':
            self.process_video(input_path, output_path, show_result)
        else:
            print(f"Error: Unsupported file format. Supported formats:")
            print("Images: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
            print("Videos: .mp4, .avi, .mov, .mkv, .flv, .wmv")

def main():
    parser = argparse.ArgumentParser(description='Human Counting System using OpenCV')
    parser.add_argument('--input', '-i', default='input/input.jpg', 
                       help='Input file path (default: input/input.jpg)')
    parser.add_argument('--output', '-o', 
                       help='Output file path (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display the result window')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('input', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Set default output path if not provided
    if not args.output:
        input_path = Path(args.input)
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            args.output = 'output/result.jpg'
        else:
            args.output = 'output/result.avi'
    
    # Initialize and run human counter
    counter = HumanCounter()
    counter.run(args.input, args.output, not args.no_display)

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import os
import requests

class HumanCounterYOLO:
    def __init__(self):
        self.net = None
        self.output_layers = None
        self.classes = []
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
    def load_model(self):
        """Load YOLOv4 model"""
        try:
            weights_path = "models/yolov4.weights"
            config_path = "models/yolov4.cfg"
            names_path = "models/coco.names"
            
            if not all(os.path.exists(p) for p in [weights_path, config_path, names_path]):
                print("❌ YOLO model files not found. Downloading...")
                if not self.download_yolo_files():
                    return False
            
            print("🔄 Loading YOLO model...")
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            with open(names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            print("✅ YOLO model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return False
    
    def download_yolo_files(self):
        """Download YOLO model files"""
        print("📥 Downloading YOLO model files...")
        
        files = {
            "models/yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            "models/coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
            "models/yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
        }
        
        os.makedirs("models", exist_ok=True)
        
        for filename, url in files.items():
            if os.path.exists(filename):
                continue
                
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
                                print(f"\r  Progress: {progress:.1f}%", end='')
                
                print(f"\n✅ Downloaded {filename}")
                
            except Exception as e:
                print(f"\n❌ Error downloading {filename}: {e}")
                return False
        
        return True
    
    def detect_people(self, frame):
        """Detect people using YOLO"""
        height, width, channels = frame.shape
        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id == 0 and confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        people_boxes = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                people_boxes.append((x, y, x + w, y + h, confidences[i]))
        
        return people_boxes
    
    def draw_detections(self, frame, people_boxes):
        """Draw bounding boxes and count"""
        people_count = len(people_boxes)
        
        for (x1, y1, x2, y2, confidence) in people_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        count_text = f"Total no. of people : {people_count}"
        cv2.putText(frame, count_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, people_count

# For compatibility with existing code
class HumanCounter(HumanCounterYOLO):
    pass
from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import base64
from scripts.main_yolo import HumanCounter
from pathlib import Path
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {
    'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'},
    'video': {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
}

# Create necessary directories
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, 'models', 'static/results']:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename, file_type=None):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type:
        return ext in ALLOWED_EXTENSIONS.get(file_type, set())
    else:
        # Check both image and video extensions
        all_extensions = set()
        for extensions in ALLOWED_EXTENSIONS.values():
            all_extensions.update(extensions)
        return ext in all_extensions

def get_file_type(filename):
    """Determine if file is image or video"""
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_EXTENSIONS['image']:
        return 'image'
    elif ext in ALLOWED_EXTENSIONS['video']:
        return 'video'
    return None

def check_model_files():
    """Check if YOLO model files exist"""
    weights_path = "models/yolov4.weights"
    config_path = "models/yolov4.cfg"
    names_path = "models/coco.names"
    return all(os.path.exists(p) for p in [weights_path, config_path, names_path])

@app.route('/')
def index():
    """Main page"""
    model_ready = check_model_files()
    return render_template('index.html', model_ready=model_ready)

@app.route('/setup')
def setup():
    """Setup page with instructions"""
    return render_template('setup.html')

@app.route('/api/check-models')
def check_models():
    """API endpoint to check if model files are ready"""
    return jsonify({'ready': check_model_files()})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
        if not check_model_files():
            return jsonify({'error': 'Model files not found. Please complete setup first.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Determine file type
        file_type = get_file_type(filename)
        
        # Process the file
        counter = HumanCounter()
        
        # Set output path
        output_filename = f"processed_{filename}"
        if file_type == 'image':
            output_filename = output_filename.rsplit('.', 1)[0] + '.jpg'
        else:
            output_filename = output_filename.rsplit('.', 1)[0] + '.mp4'
        
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Process the file
        people_count = process_file_with_counter(counter, filepath, output_path, file_type)
        
        # Copy processed file to static folder for web access
        static_output_path = os.path.join('static/results', output_filename)
        shutil.copy2(output_path, static_output_path)
        
        return jsonify({
            'success': True,
            'file_type': file_type,
            'people_count': people_count,
            'original_filename': file.filename,
            'processed_url': url_for('static', filename=f'results/{output_filename}'),
            'download_url': url_for('download_file', filename=output_filename)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_file_with_counter(counter, input_path, output_path, file_type):
    """Process file and return people count"""
    if not counter.load_model():
        raise Exception("Failed to load model")
    
    if file_type == 'image':
        # Process image
        frame = cv2.imread(input_path)
        if frame is None:
            raise Exception("Could not read image file")
        
        people_boxes = counter.detect_people(frame)
        result_frame, people_count = counter.draw_detections(frame, people_boxes)
        
        cv2.imwrite(output_path, result_frame)
        return people_count
        
    else:  # video
        # Process video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        max_people_count = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            people_boxes = counter.detect_people(frame)
            result_frame, people_count = counter.draw_detections(frame, people_boxes)
            
            max_people_count = max(max_people_count, people_count)
            out.write(result_frame)
        
        cap.release()
        out.release()
        
        return max_people_count

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed file"""
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

@app.route('/api/demo')
def create_demo():
    """Create demo files for testing"""
    try:
        from scripts.setup_demo import create_demo_structure, create_sample_image
        create_demo_structure()
        create_sample_image()
        return jsonify({'success': True, 'message': 'Demo files created successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

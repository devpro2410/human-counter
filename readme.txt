# Human Counting System

This project is a web application for detecting and counting people in images and videos using YOLOv4 and OpenCV. It provides a user-friendly interface for uploading files and visualizing detection results.

## Features

* Detects and counts people in images and videos
* Supports multiple file formats (JPG, PNG, BMP, MP4, AVI, MOV, etc.)
* Web-based interface with drag-and-drop upload
* Download processed files with detection overlays

## Project Structure

```
├── app.py                  # Main Flask web application
├── download_yolo_quick.py  # Script to quickly download YOLO model files
├── run_setup.py            # Complete setup script
├── requirements.txt        # Python dependencies
├── input/                  # Folder for input files
├── models/                 # YOLO model files (see below)
├── output/                 # Output files
├── processed/              # Processed images/videos
├── scripts/                # Supporting scripts
├── static/                 # Static files (CSS, JS, results)
├── templates/              # HTML templates
├── uploads/                # Uploaded files
```

## Datasets & Model Files

This project uses the [COCO dataset](https://cocodataset.org/) for object class names and the YOLOv4 model for detection.

**Required model files (download links):**

* [`yolov4.cfg`](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg)
* [`coco.names`](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names)
* [`yolov4.weights`](https://github.com/AlexeyAB/darknet/releases) (Download from "yolov4.weights" link)

Place all three files in the `models/` directory.

---

## 🔧 Installation

### 1. **Clone the repository**

```bash
git clone https://github.com/yourusername/human-counter-opencv.git
cd human-counter-opencv
```

### 2. **(Recommended) Create a virtual environment**

```bash
python -m venv venv
```

### 3. **Activate the virtual environment**

* On **Windows**:

  ```bash
  venv\Scripts\activate
  ```

* On **macOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

### 4. **Install dependencies**

```bash
pip install -r requirements.txt
```

### 5. **Download YOLO model files**

* Download `yolov4.cfg`, `coco.names`, and `yolov4.weights` using the links above.
* Place them in the `models/` folder.

### 6. **(Optional) Run setup script**

```bash
python run_setup.py
```

---

## 🚀 Running the Application

1. **Start the web server:**

   ```bash
   python app.py
   ```

2. **Open your browser and go to:**

   ```
   http://localhost:8080
   ```

3. **Upload an image or video to test the system.**

---

## ⚙️ Command Line Usage

You can also process files directly using the command line:

```bash
python scripts/main.py --input input/your_image.jpg
python scripts/main.py --input input/your_video.mp4
```

---

## 💡 Project Demo

* Drag and drop an image or video file onto the upload area.
* Wait for processing to complete.
* View the detection results and download the processed file.

---

## 🛠 Troubleshooting

* If you see "Setup Required" on the web page, ensure all model files are present in the `models/` folder.
* For large video files, processing may take a few minutes.

---

## 📄 License

This project is for educational and research purposes. See [LICENSE](LICENSE) for details.

---

**Credits:**
YOLOv4 by [Alexey Bochkovskiy](https://github.com/AlexeyAB/darknet)
COCO Dataset by [COCO Consortium](https://cocodataset.org/)


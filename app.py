from flask import Flask, render_template, request, jsonify, Response
import torch
from PIL import Image
import numpy as np
import cv2
import sys
import os
import pathlib
import platform
import base64
from io import BytesIO
import traceback
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- FIX 1: CROSS-PLATFORM MODEL LOADING ---
plat = platform.system()
if plat == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
elif plat == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# Add yolov5 to path
sys.path.insert(0, 'yolov5')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global model variable
model = None
model_info = {}
camera = None

def get_camera():
    """Initialize or return existing camera reference"""
    global camera
    if camera is None or not camera.isOpened():
        temp_camera = cv2.VideoCapture(0)
        if not temp_camera.isOpened():
            logger.error("Unable to access camera device")
            if temp_camera is not None:
                temp_camera.release()
            camera = None
            return None
        temp_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        temp_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        temp_camera.set(cv2.CAP_PROP_FPS, 30)
        camera = temp_camera
    return camera

def load_model():
    """Load YOLOv5 model"""
    global model, model_info
    try:
        logger.info("Starting model load...")
        from models.experimental import attempt_load
        
        device = torch.device('cpu')
        
        # Check for model file
        model_path = 'best.pt' 
        if not os.path.exists(model_path) and os.path.exists('best (1).pt'):
            model_path = 'best (1).pt'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from: {model_path}")
        model = attempt_load(model_path, device=device, fuse=False)
        model.eval()
        
        model_info = {
            'loaded': True,
            'device': str(device),
            'classes': list(model.names.values()) if isinstance(model.names, dict) else model.names,
            'stride': int(model.stride.max()) if hasattr(model, 'stride') else 32
        }
        logger.info("✓ Model loaded successfully")
        logger.info(f"Classes: {model_info['classes']}")
        return True
        
    except Exception as e:
        model_info = {
            'loaded': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        logger.error(f"✗ Model loading failed: {e}")
        logger.error(traceback.format_exc())
        return False

# Load model during startup
logger.info("Loading model during startup...")
load_model()

def process_image(image_path):
    """Process image and run detection"""
    logger.info(f"Processing image: {image_path}")
    
    if model is None:
        logger.warning("Model not loaded, attempting to load...")
        if not load_model():
            return {'success': False, 'error': 'Model failed to load. Check server logs.'}
        
    try:
        from utils.general import non_max_suppression, scale_boxes
        from utils.plots import Annotator, colors
        from utils.augmentations import letterbox
        
        # Load and preprocess image
        logger.info("Loading image...")
        image = Image.open(image_path)
        img_array = np.array(image)
        
        # Convert RGBA to RGB if needed
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        img_original = img_array.copy()
        
        # Prepare image for model
        stride = model_info.get('stride', 32)
        img = letterbox(img_array, 640, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cpu')
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        logger.info("Running inference...")
        with torch.no_grad():
            pred = model(img, augment=False, visualize=False)[0]
        
        logger.info("Applying NMS...")
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        
        detections = []
        annotator = Annotator(img_original.copy(), line_width=3, example=str(model.names))
        
        logger.info(f"Processing {len(pred)} prediction(s)...")
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_original.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    detections.append({
                        'class': model.names[c],
                        'confidence': float(conf),
                        'bbox': [float(x) for x in xyxy]
                    })
        
        annotated_img = annotator.result()
        logger.info(f"Detection complete. Found {len(detections)} object(s)")
        
        return {
            'success': True,
            'detections': detections,
            'num_detections': len(detections),
            'annotated_image': annotated_img
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        logger.error(traceback.format_exc())
        return {
            'success': False, 
            'error': str(e), 
            'traceback': traceback.format_exc()
        }

def process_frame(frame):
    """Process a single frame for live detection"""
    if model is None:
        return frame, []
        
    try:
        from utils.general import non_max_suppression, scale_boxes
        from utils.plots import Annotator, colors
        from utils.augmentations import letterbox
        
        img_original = frame.copy()
        stride = model_info.get('stride', 32)
        img = letterbox(frame, 640, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cpu')
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        with torch.no_grad():
            pred = model(img, augment=False, visualize=False)[0]
        
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        
        detections = []
        annotator = Annotator(img_original, line_width=2, example=str(model.names))
        
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_original.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    detections.append({'class': model.names[c], 'confidence': float(conf)})
        
        annotated_frame = annotator.result()
        
        has_rat = len(detections) > 0
        if has_rat:
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 60), (0, 0, 255), -1)
            cv2.putText(annotated_frame, 'HYGIENE ALERT: RAT DETECTED!', (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 60), (0, 255, 0), -1)
            cv2.putText(annotated_frame, 'HYGIENE STATUS: CLEAR', (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return annotated_frame, detections
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return frame, []

def generate_frames():
    """Stream raw camera frames without running inference to avoid frame drops"""
    while True:
        cam = get_camera()
        if cam is None:
            break
        success, frame = cam.read()
        if not success:
            logger.error("Failed to read frame from camera")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.error("Failed to encode frame for streaming")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        if camera is None or not camera.isOpened():
            break

def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    try:
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html', model_info=model_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and detection"""
    try:
        logger.info("Upload request received")
        
        # Validate request
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
            
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'success': False, 'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400
        
        # Save file
        filename = 'uploaded_' + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Process image
        logger.info("Starting image processing...")
        result = process_image(filepath)
        
        if result['success']:
            logger.info("Processing successful, preparing response...")
            
            # Load original image
            original_img = Image.open(filepath)
            original_base64 = image_to_base64(np.array(original_img))
            annotated_base64 = image_to_base64(result['annotated_image'])
            
            response_data = {
                'success': True,
                'original_image': original_base64,
                'annotated_image': annotated_base64,
                'detections': result['detections'],
                'num_detections': result['num_detections'],
                'has_rat': result['num_detections'] > 0,
                'status': 'UNHYGIENIC' if result['num_detections'] > 0 else 'CLEAR'
            }
            
            logger.info(f"Response prepared: {result['num_detections']} detections")
            return jsonify(response_data)
        else:
            logger.error(f"Processing failed: {result.get('error')}")
            return jsonify({
                'success': False, 
                'error': result.get('error', 'Unknown error')
            }), 500
            
    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Capture a single frame and run detection"""
    try:
        cam = get_camera()
        if cam is None:
            return jsonify({'success': False, 'error': 'Camera not available'}), 500
        success, frame = cam.read()
        if not success:
            logger.error("Failed to capture frame from camera")
            return jsonify({'success': False, 'error': 'Failed to capture frame'}), 500

        annotated_frame, detections = process_frame(frame)

        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        response = {
            'success': True,
            'original_image': image_to_base64(original_rgb),
            'annotated_image': image_to_base64(annotated_rgb),
            'detections': detections,
            'num_detections': len(detections),
            'has_rat': len(detections) > 0,
            'status': 'UNHYGIENIC' if detections else 'CLEAR'
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Capture frame error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'success': True})

@app.route('/model-info')
def get_model_info():
    return jsonify(model_info)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model_info.get('loaded', False),
        'model_info': model_info
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Max size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
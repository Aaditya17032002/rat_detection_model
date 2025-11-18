from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
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

# Add yolov5 to path using absolute location
BASE_DIR = pathlib.Path(__file__).resolve().parent
YOLOV5_PATH = str(BASE_DIR / 'yolov5')
if YOLOV5_PATH not in sys.path:
    sys.path.insert(0, YOLOV5_PATH)

MAX_CONTENT_LENGTH = 16 * 1024 * 1024
UPLOAD_FOLDER = str(BASE_DIR / 'uploads')
RESULTS_FOLDER = str(BASE_DIR / 'results')

app = FastAPI(title="Rat Detection Service")
templates = Jinja2Templates(directory=str(BASE_DIR / 'templates'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(BASE_DIR / 'static', exist_ok=True)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_FOLDER), name="results")

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
        model_path = BASE_DIR / 'best.pt'
        alt_model = BASE_DIR / 'best (1).pt'
        if not model_path.exists() and alt_model.exists():
            model_path = alt_model
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from: {model_path}")
        model = attempt_load(str(model_path), device=device, fuse=False)
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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_info": model_info})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle image upload and detection"""
    try:
        logger.info("Upload request received")

        if file.filename == '':
            logger.warning("Empty filename")
            raise HTTPException(status_code=400, detail='No file selected')

        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail='Invalid file type. Use JPG, JPEG, or PNG')

        contents = await file.read()
        if len(contents) > MAX_CONTENT_LENGTH:
            logger.warning("Uploaded file exceeds size limit")
            raise HTTPException(status_code=413, detail='File too large. Max size is 16MB')

        filename = 'uploaded_' + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        logger.info(f"Saving file to: {filepath}")
        with open(filepath, 'wb') as f:
            f.write(contents)

        logger.info("Starting image processing...")
        result = process_image(filepath)

        if result['success']:
            logger.info("Processing successful, preparing response...")

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
            return response_data

        logger.error(f"Processing failed: {result.get('error')}")
        raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/capture_frame")
async def capture_frame():
    """Capture a single frame and run detection"""
    try:
        cam = get_camera()
        if cam is None:
            return JSONResponse(status_code=500, content={'success': False, 'error': 'Camera not available'})
        success, frame = cam.read()
        if not success:
            logger.error("Failed to capture frame from camera")
            return JSONResponse(status_code=500, content={'success': False, 'error': 'Failed to capture frame'})

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
        return response
    except Exception as e:
        logger.error(f"Capture frame error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})

@app.post("/stop_camera")
async def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return {'success': True}

@app.get("/model-info")
async def get_model_info():
    return model_info

@app.get("/health")
async def health_check():
    return {
        'status': 'healthy',
        'model_loaded': model_info.get('loaded', False),
        'model_info': model_info
    }

if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)
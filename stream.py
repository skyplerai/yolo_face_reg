import cv2
import numpy as np
from flask import Flask, Response, render_template
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
import os
from datetime import datetime, date
import time
import logging
import threading
import queue
from pymongo import MongoClient
import torch
from ultralytics import YOLO

# Set up logging
# This configures the logging level. You can change it to logging.DEBUG for more verbose output
# or logging.WARNING for less output.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MongoDB setup
# If you change the MongoDB connection string, make sure it points to your database server
# The database name 'face_recognition_db' and collection name 'temp_faces' can be modified
# to suit your needs
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
temp_faces = db['temp_faces']

# RTSP URL
# Change this URL to connect to a different RTSP stream
rtsp_url = 'rtsp://admin:skypler@sriram@210.18.176.33:1024/Streaming/channels/101'
cap = None

# Initialize YOLO model
# The device selection automatically uses CUDA if available, otherwise CPU
# You can force CPU usage by changing this to device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Change 'yolov8m-face.pt' to use a different YOLO model file
facemodel = YOLO('yolov8m-face.pt').to(device)
print(device)

# Initialize DeepSORT
# Adjust these parameters to fine-tune tracking performance:
# - Increase max_cosine_distance for more lenient matching
# - Decrease it for stricter matching
# - Increase nn_budget to consider more past detections (may slow down tracking)
max_cosine_distance = 0.3
nn_budget = 300
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# Increase max_age to keep lost tracks for longer, decrease for quicker removal of lost tracks
tracker = Tracker(metric, max_age=100)

# Queues for parallel processing
# Increase maxsize for more buffering (may increase latency), decrease for lower latency but potential frame drops
frame_queue = queue.Queue(maxsize=30)
result_queue = queue.Queue(maxsize=30)

# Flag to signal thread termination
terminate_flag = False

# Global variables for face ID management
current_date = date.today()
face_id_counter = 1
face_id_mapping = {}
frame_save_counter = {}

def ensure_connection(max_retries=10):
    # This function attempts to establish a connection to the RTSP stream
    # Increase max_retries for more connection attempts, decrease for quicker timeout
    global cap
    for attempt in range(max_retries):
        if cap is None or not cap.isOpened():
            logger.info(f"Attempting to connect to RTSP stream (Attempt {attempt + 1}/{max_retries})...")
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            if cap.isOpened():
                logger.info("Successfully connected to RTSP stream.")
                return True
        time.sleep(2)  # Wait 2 seconds between attempts, adjust as needed
    logger.error("Failed to connect to RTSP stream after multiple attempts.")
    return False

def read_frame():
    # This function reads a frame from the RTSP stream
    # It includes error handling and reconnection logic
    global cap
    if cap is None or not cap.isOpened():
        if not ensure_connection():
            return None
    ret, frame = cap.read()
    if not ret:
        logger.warning("Failed to read frame. Reconnecting...")
        if ensure_connection():
            ret, frame = cap.read()
    return frame if ret else None

def detect_faces(frame):
    # This function detects faces in a frame using the YOLO model
    # Convert frame to RGB (YOLO expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLO detection
    # Adjust the confidence threshold (conf=0.49) to change detection sensitivity
    results = facemodel(frame_rgb, conf=0.49)
    
    # Extract face detections
    faces = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf.item()
            faces.append([x1, y1, x2 - x1, y2 - y1, confidence])
    
    return np.array(faces)

def generate_simple_feature(face, frame):
    # This function generates a simple feature vector for a detected face
    # It's a basic implementation and could be replaced with a more sophisticated feature extractor
    x, y, w, h, _ = face.astype(int)
    face_roi = frame[y:y+h, x:x+w]
    if face_roi.size == 0:
        return np.zeros(64*64)
    # Resize to 64x64 - adjust these dimensions to change the feature vector size
    face_roi = cv2.resize(face_roi, (64, 64))
    feature = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).flatten()
    norm = np.linalg.norm(feature)
    return feature / norm if norm != 0 else feature

def get_next_face_id():
    # This function generates unique face IDs
    # It resets the counter each day to avoid excessively large numbers
    global face_id_counter, current_date
    
    # Check if date has changed
    today = date.today()
    if today != current_date:
        current_date = today
        face_id_counter = 1
        face_id_mapping.clear()
    
    face_id = f"unknown_{face_id_counter:03d}"
    face_id_counter += 1
    return face_id

def save_face_image(frame, track):
    # This function saves detected face images and updates the database
    global face_id_mapping, frame_save_counter
    
    track_id = int(track.track_id)
    if track_id not in face_id_mapping:
        face_id_mapping[track_id] = get_next_face_id()
        frame_save_counter[track_id] = 0
    
    frame_save_counter[track_id] += 1
    # Only save every 7th frame - adjust this value to change save frequency
    if frame_save_counter[track_id] % 7 != 0:
        return
    
    face_id = face_id_mapping[track_id]
    today = current_date.strftime("%Y-%m-%d")
    
    # Directory structure: YYYY-MM-DD/unknown_XXX/images/
    directory = os.path.join(today, face_id, "images")
    os.makedirs(directory, exist_ok=True)
    
    bbox = track.to_tlbr()
    h, w = frame.shape[:2]
    
    # Add padding to include chin, hair, and ears
    # Adjust these padding values to change the saved face image size
    pad_w = 0.2 * (bbox[2] - bbox[0])
    pad_h = 0.2 * (bbox[3] - bbox[1])
    x1, y1 = max(0, int(bbox[0] - pad_w)), max(0, int(bbox[1] - pad_h))
    x2, y2 = min(w, int(bbox[2] + pad_w)), min(h, int(bbox[3] + pad_h))
    
    face_img = frame[y1:y2, x1:x2]
    
    if face_img.size > 0:  # Only save if the face image is not empty
        image_count = len([f for f in os.listdir(directory) if f.endswith('.jpg')])
        # Maximum of 15 images per face - adjust this limit as needed
        if image_count < 15:
            filename = os.path.join(directory, f"{face_id}_{image_count:02d}.jpg")
            cv2.imwrite(filename, face_img)
            
            # Update MongoDB
            # This upsert operation creates or updates the document for each face
            temp_faces.update_one(
                {'face_id': face_id},
                {'$push': {'image_paths': filename},
                 '$setOnInsert': {'timestamp': datetime.now(), 'processed': False}},
                upsert=True
            )
    
    return face_id

def process_frame(frame):
    # This function processes each frame, detecting and tracking faces
    faces = detect_faces(frame)
    detections = [Detection(face[:4], face[4], generate_simple_feature(face, frame)) for face in faces]
    
    tracker.predict()
    tracker.update(detections)
    
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        face_id = save_face_image(frame, track)
        # Draw bounding box and face ID on the frame
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(frame, face_id, (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

def frame_producer():
    # This function continuously reads frames from the RTSP stream
    while not terminate_flag:
        frame = read_frame()
        if frame is not None:
            if frame_queue.full():
                frame_queue.get()  # Remove oldest frame if queue is full
            frame_queue.put(frame)
        else:
            time.sleep(0.1)  # Short sleep to prevent CPU overuse if frames are not available

def frame_consumer():
    # This function processes frames from the queue
    while not terminate_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()
            processed_frame = process_frame(frame)
            if result_queue.full():
                result_queue.get()  # Remove oldest processed frame if queue is full
            result_queue.put(processed_frame)
        else:
            time.sleep(0.01)  # Short sleep to prevent CPU overuse if no frames are available

def generate_frames():
    # This function yields processed frames for the Flask video feed
    while True:
        if not result_queue.empty():
            frame = result_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.01)  # Short sleep to prevent CPU overuse if no frames are available

@app.route('/')
def index():
    # Route for the main page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Route for the video feed
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    ensure_connection()
    
    # Start producer and consumer threads
    producer_thread = threading.Thread(target=frame_producer)
    consumer_thread = threading.Thread(target=frame_consumer)
    producer_thread.start()
    consumer_thread.start()
    
    try:
        # Run the Flask app
        # Change host and port as needed. debug=True enables debug mode
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        # Ensure clean shutdown
        terminate_flag = True
        producer_thread.join()
        consumer_thread.join()
        if cap is not None:
            cap.release()

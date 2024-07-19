from pymongo import MongoClient
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
temp_faces = db['temp_faces']
perm_faces = db['perm_faces']

def update_face_id():
    """Update a temporary face ID to a permanent one if it has encodings."""
    unknown_id = input("Enter the unknown ID to update: ")
    new_name = input("Enter the new name: ")

    temp_face = temp_faces.find_one({'face_id': unknown_id})
    if temp_face:
        if 'embeddings' in temp_face and temp_face['embeddings']:
            # Move data to permanent collection
            perm_faces.insert_one({
                'name': new_name,
                'embeddings': temp_face['embeddings'],
                'image_paths': temp_face['image_paths'],
                'last_seen': temp_face.get('timestamp', datetime.now())
            })
            
            # Remove from temporary collection
            temp_faces.delete_one({'_id': temp_face['_id']})
            
            logging.info(f"Successfully updated '{unknown_id}' to '{new_name}' and moved to permanent faces")
        else:
            logging.warning(f"No encodings found for '{unknown_id}'. Try again later.")
    else:
        logging.warning(f"No face found for ID '{unknown_id}'")

def run_face_id_editor():
    """Run the face ID editor interface."""
    while True:
        update_face_id()
        if input("Do you want to update another face ID? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    run_face_id_editor()


    
# import cv2
# import numpy as np
# from flask import Flask, Response, render_template
# from deep_sort_realtime.deep_sort import nn_matching
# from deep_sort_realtime.deep_sort.detection import Detection
# from deep_sort_realtime.deep_sort.tracker import Tracker
# import os
# from datetime import datetime
# import time
# import logging
# import threading
# import queue
# from pymongo import MongoClient

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # MongoDB setup
# client = MongoClient('mongodb://localhost:27017/')
# db = client['face_recognition_db']
# temp_faces = db['temp_faces']

# # Initialize face detector
# face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# # Check if CUDA is available and set it as the target
# if cv2.cuda.getCudaEnabledDeviceCount() > 0:
#     face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#     logger.info("Using CUDA for face detection")
#     device = "GPU"
# else:
#     logger.info("CUDA is not available. Using CPU for face detection.")
#     device = "CPU"

# # RTSP URL
# rtsp_url = 'rtsp://admin:skypler@sriram@210.18.176.33:554/Streaming/Channels/101'
# cap = None

# # Initialize DeepSORT
# max_cosine_distance = 0.3
# nn_budget = None
# metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# tracker = Tracker(metric)

# # Queues for parallel processing
# frame_queue = queue.Queue(maxsize=30)
# result_queue = queue.Queue(maxsize=30)

# # Flag to signal thread termination
# terminate_flag = False

# def ensure_connection(max_retries=10):
#     global cap
#     for attempt in range(max_retries):
#         if cap is None or not cap.isOpened():
#             logger.info(f"Attempting to connect to RTSP stream (Attempt {attempt + 1}/{max_retries})...")
#             if cap is not None:
#                 cap.release()
#             cap = cv2.VideoCapture(rtsp_url)
#             if cap.isOpened():
#                 logger.info("Successfully connected to RTSP stream.")
#                 return True
#         time.sleep(2)
#     logger.error("Failed to connect to RTSP stream after multiple attempts.")
#     return False

# def read_frame():
#     global cap
#     if cap is None or not cap.isOpened():
#         if not ensure_connection():
#             return None
#     ret, frame = cap.read()
#     if not ret:
#         logger.warning("Failed to read frame. Reconnecting...")
#         if ensure_connection():
#             ret, frame = cap.read()
#     return frame if ret else None

# def detect_faces(frame):
#     print(f"Detecting faces using {device}...")
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     face_detector.setInput(blob)
#     detections = face_detector.forward()
    
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             faces.append([startX, startY, endX - startX, endY - startY, confidence])
    
#     print(f"Detected {len(faces)} faces")
#     return np.array(faces)

# def generate_simple_feature(face, frame):
#     x, y, w, h, _ = face.astype(int)
#     face_roi = frame[y:y+h, x:x+w]
#     if face_roi.size == 0:
#         return np.zeros(64*64)
#     face_roi = cv2.resize(face_roi, (64, 64))
#     feature = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).flatten()
#     norm = np.linalg.norm(feature)
#     return feature / norm if norm != 0 else feature

# def save_face_image(frame, track):
#     today = datetime.now().strftime("%Y-%m-%d")
#     face_id = f"unknown_{int(track.track_id):03d}"
    
#     directory = os.path.join(today, face_id, "images")
#     os.makedirs(directory, exist_ok=True)
    
#     bbox = track.to_tlbr()
#     h, w = frame.shape[:2]
    
#     padding_factor = 0.2
#     width = bbox[2] - bbox[0]
#     height = bbox[3] - bbox[1]
#     pad_w, pad_h = int(width * padding_factor), int(height * padding_factor)
    
#     x1, y1 = max(0, int(bbox[0] - pad_w)), max(0, int(bbox[1] - pad_h))
#     x2, y2 = min(w, int(bbox[2] + pad_w)), min(h, int(bbox[3] + pad_h))
    
#     face_img = frame[y1:y2, x1:x2]
    
#     image_count = len([f for f in os.listdir(directory) if f.endswith('.jpg')])
#     if image_count < 15:
#         filename = os.path.join(directory, f"{face_id}_{image_count:02d}.jpg")
#         cv2.imwrite(filename, face_img)
        
#         # Update MongoDB
#         temp_faces.update_one(
#             {'face_id': face_id},
#             {'$push': {'image_paths': filename},
#              '$setOnInsert': {'timestamp': datetime.now(), 'processed': False}},
#             upsert=True
#         )

# def process_frame(frame):
#     print("Processing frame...")
#     faces = detect_faces(frame)
#     detections = [Detection(face[:4], face[4], generate_simple_feature(face, frame)) for face in faces]
    
#     tracker.predict()
#     tracker.update(detections)
    
#     for track in tracker.tracks:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue
#         bbox = track.to_tlbr()
#         face_id = f"unknown_{int(track.track_id):03d}"
#         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
#         cv2.putText(frame, face_id, (int(bbox[0]), int(bbox[1]) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
#         save_face_image(frame, track)
    
#     print("Frame processing completed")
#     return frame

# def frame_producer():
#     while not terminate_flag:
#         frame = read_frame()
#         if frame is not None:
#             if frame_queue.full():
#                 frame_queue.get()
#             frame_queue.put(frame)
#         else:
#             time.sleep(0.1)

# def frame_consumer():
#     while not terminate_flag:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             processed_frame = process_frame(frame)
#             if result_queue.full():
#                 result_queue.get()
#             result_queue.put(processed_frame)
#         else:
#             time.sleep(0.01)

# def generate_frames():
#     while True:
#         if not result_queue.empty():
#             frame = result_queue.get()
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         else:
#             time.sleep(0.01)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     ensure_connection()
    
#     producer_thread = threading.Thread(target=frame_producer)
#     consumer_thread = threading.Thread(target=frame_consumer)
#     producer_thread.start()
#     consumer_thread.start()
    
#     try:
#         app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
#     finally:
#         terminate_flag = True
#         producer_thread.join()
#         consumer_thread.join()
#         if cap is not None:
#             cap.release()






# import face_recognition
# import cv2
# import os
# import numpy as np
# from datetime import datetime, timedelta
# from pymongo import MongoClient
# import time

# # MongoDB setup
# client = MongoClient('mongodb://localhost:27017/')
# db = client['face_recognition_db']
# temp_faces = db['temp_faces']
# perm_faces = db['perm_faces']

# def get_face_embedding(image_path):
#     image = face_recognition.load_image_file(image_path)
#     face_encodings = face_recognition.face_encodings(image)
#     return face_encodings[0] if face_encodings else None

# def perform_face_recognition():
#     unprocessed_faces = temp_faces.find({'processed': False})
#     permanent_faces = list(perm_faces.find())
    
#     for temp_face in unprocessed_faces:
#         temp_embeddings = [get_face_embedding(path) for path in temp_face['image_paths'][4:11]]
#         temp_embeddings = [emb for emb in temp_embeddings if emb is not None]
        
#         if not temp_embeddings:
#             continue
        
#         best_match = None
#         best_match_distance = float('inf')
        
#         for perm_face in permanent_faces:
#             perm_embeddings = perm_face['embeddings']
#             distances = face_recognition.face_distance(perm_embeddings, temp_embeddings)
#             avg_distance = np.mean(distances)
            
#             if avg_distance < best_match_distance and avg_distance < 0.6:
#                 best_match = perm_face
#                 best_match_distance = avg_distance
        
#         if best_match:
#             # Update permanent face record
#             perm_faces.update_one(
#                 {'_id': best_match['_id']},
#                 {'$set': {'last_seen': datetime.now()},
#                  '$push': {'embeddings': {'$each': temp_embeddings},
#                            'image_paths': {'$each': temp_face['image_paths']}}}
#             )
#             # Remove temporary face record
#             temp_faces.delete_one({'_id': temp_face['_id']})
#         else:
#             # Mark as processed but keep the record
#             temp_faces.update_one({'_id': temp_face['_id']}, {'$set': {'processed': True}})

# def main_loop():
#     last_recognition_time = datetime.now() - timedelta(minutes=20)
    
#     while True:
#         current_time = datetime.now()
#         if (current_time - last_recognition_time).total_seconds() >= 1200:  # 20 minutes
#             perform_face_recognition()
#             last_recognition_time = current_time
        
#         time.sleep(60)  # Check every minute

# if __name__ == "__main__":
#     main_loop()





# from pymongo import MongoClient
# import os

# # MongoDB setup
# client = MongoClient('mongodb://localhost:27017/')
# db = client['face_recognition_db']
# temp_faces = db['temp_faces']
# perm_faces = db['perm_faces']

# def update_face_id():
#     unknown_id = input("Enter the unknown ID to update: ")
#     new_name = input("Enter the new name: ")

#     temp_face = temp_faces.find_one({'face_id': unknown_id})
#     if temp_face:
#         # Move data to permanent collection
#         perm_faces.insert_one({
#             'name': new_name,
#             'embeddings': temp_face.get('embeddings', []),
#             'image_paths': temp_face['image_paths'],
#             'last_seen': temp_face['timestamp']
#         })
        
#         # Remove from temporary collection
#         temp_faces.delete_one({'_id': temp_face['_id']})
        
#         print(f"Successfully updated '{unknown_id}' to '{new_name}'")
#     else:
#         print(f"No face encodings found for ID '{unknown_id}'")

# if __name__ == "__main__":
#     update_face_id()
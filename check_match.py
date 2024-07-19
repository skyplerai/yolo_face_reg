import os
import time
from datetime import datetime, timedelta
import schedule
import face_recognition
import cv2
import numpy as np
from pymongo import MongoClient
import logging
import torch
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
temp_faces = db['temp_faces']
perm_faces = db['perm_faces']
analytics = db['analytics']

# Face recognition settings
FACE_MATCH_THRESHOLD = 0.6
MIN_IMAGES_TO_PROCESS = 5
MAX_IMAGES_TO_PROCESS = 10
BLUR_THRESHOLD = 0
FACE_ANGLE_THRESHOLD = 30

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using {device} for processing.")

def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_face_angle(face_landmarks):
    left_eye = np.mean(face_landmarks['left_eye'], axis=0)
    right_eye = np.mean(face_landmarks['right_eye'], axis=0)
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    return abs(angle)

def get_face_embedding(image_path):
    logger.info(f"Processing image: {image_path}")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
        
        if detect_blur(image) < BLUR_THRESHOLD:
            logger.info(f"Image too blurry: {image_path}")
            return None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image, model="cnn", number_of_times_to_upsample=0)
        
        if not face_locations:
            logger.info(f"No face detected in image: {image_path}")
            return None
        
        face_landmarks = face_recognition.face_landmarks(rgb_image, face_locations)[0]
        face_angle = detect_face_angle(face_landmarks)
        
        if face_angle > FACE_ANGLE_THRESHOLD:
            logger.info(f"Face angle too extreme: {image_path}")
            return None
        
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=1, model="large")[0]
        return face_encoding
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def process_unprocessed_faces():
    logger.info("Starting to process unprocessed faces")
    
    unprocessed_faces = list(temp_faces.find({'processed': False}))
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for face in unprocessed_faces:
            logger.info(f"Processing face: {face['face_id']}")
            
            image_paths = face['image_paths']
            num_images = min(max(MIN_IMAGES_TO_PROCESS, len(image_paths) // 2), MAX_IMAGES_TO_PROCESS)
            selected_images = image_paths[len(image_paths)//4 : 3*len(image_paths)//4][:num_images]
            
            embeddings = []
            for result in executor.map(get_face_embedding, selected_images):
                if result is not None and not isinstance(result, np.ndarray):
                    logger.warning(f"Unexpected result type from get_face_embedding: {type(result)}")
                elif result is not None:
                    embeddings.append(result.tolist())
            
            if embeddings:
                temp_faces.update_one(
                    {'_id': face['_id']},
                    {
                        '$set': {
                            'embeddings': embeddings,
                            'processed': True,
                            'last_processed': datetime.now()
                        }
                    }
                )
                logger.info(f"Updated embeddings for face: {face['face_id']}")
            else:
                logger.warning(f"No valid embeddings generated for face: {face['face_id']}")

def match_faces():
    logger.info("Starting face matching process")
    
    processed_faces = list(temp_faces.find({'processed': True, 'matched': {'$ne': True}}))
    perm_face_list = list(perm_faces.find())
    
    matched_faces = []
    
    for face in processed_faces:
        logger.info(f"Matching face: {face['face_id']}")
        
        best_match = None
        best_match_distance = float('inf')
        
        for perm_face in perm_face_list:
            temp_embeddings = np.array(face['embeddings'])
            perm_embeddings = np.array(perm_face['embeddings'])
            
            distances = face_recognition.face_distance(perm_embeddings, temp_embeddings)
            min_distance = np.min(distances)
            
            if min_distance < best_match_distance and min_distance < FACE_MATCH_THRESHOLD:
                best_match = perm_face
                best_match_distance = min_distance
        
        if best_match:
            logger.info(f"Match found for {face['face_id']}: {best_match['name']}")
            
            temp_faces.update_one(
                {'_id': face['_id']},
                {
                    '$set': {
                        'matched': True,
                        'matched_name': best_match['name'],
                        'matched_id': str(best_match['_id'])
                    }
                }
            )
            
            perm_faces.update_one(
                {'_id': best_match['_id']},
                {
                    '$push': {
                        'image_paths': {'$each': face['image_paths']},
                        'embeddings': {'$each': face['embeddings']}
                    },
                    '$set': {'last_seen': datetime.now()}
                }
            )
            
            matched_faces.append(best_match['name'])
        else:
            logger.info(f"No match found for {face['face_id']}")
            
        # Mark as matched even if no match found to avoid reprocessing
        temp_faces.update_one({'_id': face['_id']}, {'$set': {'matched': True}})
    
    return matched_faces

def generate_analytics_report(matched_faces):
    if matched_faces:
        # Get the start of the current day
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Count occurrences of each person
        person_counts = Counter(matched_faces)
        
        # Check if there's an existing report for today
        existing_report = analytics.find_one({'date': today})
        
        if existing_report:
            # Update existing report
            for person, count in person_counts.items():
                analytics.update_one(
                    {'_id': existing_report['_id']},
                    {
                        '$inc': {f'detected_persons.{person}': count},
                        '$set': {'last_updated': datetime.now()}
                    }
                )
            logger.info("Updated existing analytics report")
        else:
            # Create new report
            report = {
                'date': today,
                'detected_persons': dict(person_counts),
                'total_detections': len(matched_faces),
                'last_updated': datetime.now()
            }
            analytics.insert_one(report)
            logger.info("Created new analytics report")
        
        # Log the current state
        current_report = analytics.find_one({'date': today})
        logger.info("Analytics Report:")
        logger.info(f"Date: {current_report['date']}")
        logger.info(f"Detected Persons: {current_report['detected_persons']}")
        logger.info(f"Total Detections: {sum(current_report['detected_persons'].values())}")
        logger.info(f"Last Updated: {current_report['last_updated']}")
    else:
        logger.info("No faces matched in this session.")

def perform_extraction(duration_minutes=5):
    logger.info(f"Starting extraction process for {duration_minutes} minutes...")
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    while datetime.now() < end_time:
        process_unprocessed_faces()
    logger.info("Extraction process completed.")

def perform_matching(duration_minutes=2):
    logger.info(f"Starting matching process for {duration_minutes} minutes...")
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    matched_faces = []
    while datetime.now() < end_time:
        matched_faces.extend(match_faces())
    generate_analytics_report(matched_faces)
    logger.info("Matching process completed.")

if __name__ == "__main__":
    # Schedule extraction and matching tasks
    schedule.every(2).minutes.do(perform_extraction)
    schedule.every(2).minutes.do(perform_matching)
    
    logger.info("Face recognition system started. Running continuously...")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

# Effects of changing values:
# 1. FACE_MATCH_THRESHOLD:
#    - Increase: More lenient matching, potentially more false positives
#    - Decrease: Stricter matching, potentially more false negatives
#
# 2. MIN_IMAGES_TO_PROCESS / MAX_IMAGES_TO_PROCESS:
#    - Increase: More comprehensive analysis, but slower processing
#    - Decrease: Faster processing, but potentially less accurate results
#
# 3. BLUR_THRESHOLD:
#    - Increase: More lenient on image quality, potentially processing more blurry images
#    - Decrease: Stricter on image quality, potentially discarding more images
#
# 4. FACE_ANGLE_THRESHOLD:
#    - Increase: More lenient on face angles, potentially processing more off-angle faces
#    - Decrease: Stricter on face angles, potentially discarding more images
#
# 5. duration_minutes in perform_extraction and perform_matching:
#    - Increase: Longer processing time, potentially more comprehensive results
#    - Decrease: Shorter processing time, potentially less comprehensive results
#
# 6. schedule.every(2).minutes:
#    - Increase: Less frequent processing, potentially missing some real-time updates
#    - Decrease: More frequent processing, potentially causing system overload
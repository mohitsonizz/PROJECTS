import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
import time
import winsound
import math  # Distance calculate karne ke liye naya module

# Apna model aur Dlib predictors load karein
model = load_model('blink_model.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

cap = cv2.VideoCapture(0)
if not cap.isOpened() or not cap.read()[0]:
    cap = cv2.VideoCapture(1)

# --- Variables ---
blink_count = 0
state = "Open"
eye_closed_start_time = None
ALARM_DURATION = 2.0  

# Yawning ke naye variables
yawn_count = 0
is_yawning = False
YAWN_THRESH = 0.6  # Agar MAR isse zyada hua toh matlab driver ubaasi le raha hai

# Aankh crop karne ka function
def crop_eye(frame, landmarks, points):
    x_coords = [landmarks.part(i).x for i in points]
    y_coords = [landmarks.part(i).y for i in points]
    
    x_min, x_max = min(x_coords) - 5, max(x_coords) + 5
    y_min, y_max = min(y_coords) - 5, max(y_coords) + 5
    x_min, y_min = max(0, x_min), max(0, y_min)
    
    eye_crop = frame[y_min:y_max, x_min:x_max]
    return eye_crop, (x_min, y_min, x_max, y_max)

# --- NAYA FUNCTION: Mouth Aspect Ratio (MAR) nikalne ke liye ---
def calculate_mar(landmarks):
    # Upar aur neeche ke hothon (lips) ke points
    p50 = (landmarks.part(50).x, landmarks.part(50).y)
    p51 = (landmarks.part(51).x, landmarks.part(51).y)
    p52 = (landmarks.part(52).x, landmarks.part(52).y)
    
    p58 = (landmarks.part(58).x, landmarks.part(58).y)
    p57 = (landmarks.part(57).x, landmarks.part(57).y)
    p56 = (landmarks.part(56).x, landmarks.part(56).y)
    
    # Munh ke left aur right corners
    p48 = (landmarks.part(48).x, landmarks.part(48).y) 
    p54 = (landmarks.part(54).x, landmarks.part(54).y) 
    
    # Vertical distance (Munh kitna khula hai)
    d1 = math.dist(p50, p58)
    d2 = math.dist(p51, p57)
    d3 = math.dist(p52, p56)
    
    # Horizontal distance (Munh kitna chauda hai)
    d4 = math.dist(p48, p54)
    
    # MAR Formula
    mar = (d1 + d2 + d3) / (3.0 * d4)
    return mar

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        try:
            # 1. Aankh ka kaam (Blink & Drowsiness)
            left_eye_img, left_box = crop_eye(gray, landmarks, range(36, 42))
            right_eye_img, right_box = crop_eye(gray, landmarks, range(42, 48))

            l_eye = cv2.resize(left_eye_img, (24, 24)) / 255.0
            l_eye = l_eye.reshape(1, 24, 24, 1)
            l_pred = model.predict(l_eye, verbose=0)[0][0]
            
            r_eye = cv2.resize(right_eye_img, (24, 24)) / 255.0
            r_eye = r_eye.reshape(1, 24, 24, 1)
            r_pred = model.predict(r_eye, verbose=0)[0][0]
            
            cv2.rectangle(frame, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (255, 0, 0), 1)
            cv2.rectangle(frame, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (255, 0, 0), 1)
            
            if l_pred < 0.15 and r_pred < 0.15:
                state = "Closed"
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                else:
                    closed_duration = time.time() - eye_closed_start_time
                    if closed_duration >= ALARM_DURATION:
                        cv2.putText(frame, "ALERT! DROWSINESS DETECTED!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        winsound.Beep(2500, 300) 
            else:
                state = "Open"
                if eye_closed_start_time is not None:
                    closed_duration = time.time() - eye_closed_start_time
                    if closed_duration < ALARM_DURATION:
                        blink_count += 1
                    eye_closed_start_time = None
                
            # 2. Munh ka kaam (Yawning Detection)
            mar = calculate_mar(landmarks)
            
            # Munh ke upar rectangle banayein
            mouth_x = [landmarks.part(i).x for i in range(48, 68)]
            mouth_y = [landmarks.part(i).y for i in range(48, 68)]
            cv2.rectangle(frame, (min(mouth_x)-5, min(mouth_y)-5), (max(mouth_x)+5, max(mouth_y)+5), (0, 255, 255), 1)

            if mar > YAWN_THRESH:
                is_yawning = True
                cv2.putText(frame, "YAWNING DETECTED!", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            else:
                if is_yawning: # Agar munh khula tha aur ab band ho gaya
                    yawn_count += 1
                    is_yawning = False

            # Screen par details dikhayein
            cv2.putText(frame, f"State: {state} | MAR: {mar:.2f}", (face.left(), face.top()-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            pass
            
    cv2.putText(frame, f"Total Blinks: {blink_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Total Yawns: {yawn_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Multi-Modal Driver Drowsiness System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()